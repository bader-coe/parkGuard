"""
ParkGuard - trigger-driven local gate camera + SQLite face/permit verification.

Design goals
------------
* The camera stays in standby at startup. It never opens until the ESP32
  ultrasonic sensor sends an explicit HTTP trigger (``/start_camera`` or
  ``/trigger``). Each trigger runs exactly one scan session, then the camera is
  released and the system returns to standby.
* Authorized faces get a green box with the real name. Unknown or unauthorized
  faces get a red box labelled "Unknown"; rendering and SQLite logging use the
  same decision so they can never disagree.
* Every accepted event snapshot is written to ./capture_logs with a
  microsecond-precision timestamp and recorded in SQLite.
* Face encodings and permits are both read from the local SQLite database.
  Permits must be exactly four digits:
      1xxx -> resident, upper floor
      0xxx -> visitor, ground floor
* Unauthorized attempts accumulate in SQLite. The first attempt only logs; the
  second and later attempts also sound the ESP32 buzzer and set lockdown.
  A successful authorized scan (or /reset_security) clears the streak and the
  lockdown.
* No cloud service is used. The only network calls are optional, local HTTP
  calls to an ESP32 gate controller.

Run:
    python main.py
    python main.py --rebuild-encodings
    python main.py --camera 1 --no-preview
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template_string, request, send_from_directory

import esp32_comm
# face_recognition is a lazy proxy: the dlib model files are opened on the
# first recognition call, not when main.py is imported.
from face_models import face_recognition, missing_model_files

try:  # pyttsx3 is optional; a Windows SAPI fallback keeps local audio working.
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover - depends on the local environment
    pyttsx3 = None  # type: ignore[assignment]


# =============================================================================
# Paths and configuration
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Load a local .env file when python-dotenv is installed. Existing environment
# variables always win, which keeps tests and container overrides predictable.
try:  # pragma: no cover - depends on the local environment
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:  # pragma: no branch - trivial environment check
    load_dotenv(PROJECT_DIR / ".env")


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, resolving relative paths from the project."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DB_PATH = _env_path("PARKGUARD_DB_PATH", PROJECT_DIR / "park_guard.db")
CAPTURE_DIR = _env_path("PARKGUARD_CAPTURE_DIR", PROJECT_DIR / "capture_logs")
_DEFAULT_DATASET = (
    PROJECT_DIR / "faces"
    if (PROJECT_DIR / "faces").is_dir()
    else PROJECT_DIR / "dataset_faces"
)
DATASET_DIR = _env_path("PARKGUARD_DATASET_DIR", _DEFAULT_DATASET)
ENCODINGS_PICKLE = _env_path("PARKGUARD_ENCODINGS_PICKLE", PROJECT_DIR / "encodings.pickle")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = _env_int("PARKGUARD_CAMERA_INDEX", 0)
CAMERA_WIDTH = _env_int("PARKGUARD_CAMERA_WIDTH", 1280)
CAMERA_HEIGHT = _env_int("PARKGUARD_CAMERA_HEIGHT", 720)
CAMERA_RECONNECT_SECONDS = max(0.2, _env_float("PARKGUARD_CAMERA_RECONNECT_SECONDS", 1.5))

# Processing / recognition tuning.
FRAME_SCALE = min(1.0, max(0.1, _env_float("PARKGUARD_FRAME_SCALE", 0.4)))
FACE_TOLERANCE = min(1.0, max(0.1, _env_float("PARKGUARD_FACE_TOLERANCE", 0.45)))
DB_REFRESH_SECONDS = max(0.0, _env_float("PARKGUARD_DB_REFRESH_SECONDS", 15.0))

# A clear, valid face must stay continuously visible for this long before an
# access event is captured. The countdown is paused whenever the face is
# missing, covered, too small, or otherwise unclear.
STABILITY_SECONDS = max(0.0, _env_float("PARKGUARD_STABILITY_SECONDS", 3.0))
# Consecutive observations farther apart than this are treated as a gap. Time
# spent while the face was absent is never added to the stability countdown.
STABILITY_FRAME_GAP_SECONDS = max(
    0.05, _env_float("PARKGUARD_STABILITY_FRAME_GAP_SECONDS", 0.5)
)
# A face smaller than this (in original-frame pixels) is not considered clear.
MIN_CLEAR_FACE_PIXELS = max(16, _env_int("PARKGUARD_MIN_CLEAR_FACE_PIXELS", 48))
# Reject flat or very dark crops; these usually mean a hand/object covers the
# face or the camera is obstructed.
MIN_CLEAR_FACE_CONTRAST = max(0.0, _env_float("PARKGUARD_MIN_CLEAR_FACE_CONTRAST", 8.0))
# How long a face may be briefly missed before its track is discarded.
TRACK_TTL_SECONDS = max(0.2, _env_float("PARKGUARD_TRACK_TTL_SECONDS", 1.5))
IOU_MATCH_THRESHOLD = 0.25
CENTER_MATCH_RATIO = 0.75

# One ESP32 ultrasonic trigger runs one bounded scan session.
SCAN_SESSION_TIMEOUT_SECONDS = max(1.0, _env_float("PARKGUARD_SCAN_TIMEOUT_SECONDS", 10.0))
SCAN_IDLE_POLL_SECONDS = max(0.02, _env_float("PARKGUARD_SCAN_IDLE_POLL_SECONDS", 0.2))

# The first unauthorized attempt only logs. From this attempt onward the
# ESP32 buzzer sounds and lockdown is set.
BUZZER_AFTER_ATTEMPTS = max(1, _env_int("PARKGUARD_BUZZER_AFTER_ATTEMPTS", 2))

JPEG_QUALITY = min(100, max(10, _env_int("PARKGUARD_JPEG_QUALITY", 92)))

# The server never stores a local board address in source control. Set
# PARKGUARD_ESP32_IP (or pass --esp32-ip) to the ESP32's LAN address, for
# example 192.0.2.10. An empty value disables all ESP32 HTTP calls.
ESP32_IP = os.environ.get("PARKGUARD_ESP32_IP", "").strip()
ESP32_TIMEOUT_SECONDS = max(0.2, _env_float("PARKGUARD_ESP32_TIMEOUT_SECONDS", 1.5))
ESP32_ENDPOINTS = {
    "authorized": os.environ.get("PARKGUARD_ESP32_AUTHORIZED_PATH", "authorized"),
    "unauthorized": os.environ.get("PARKGUARD_ESP32_UNAUTHORIZED_PATH", "unauthorized"),
    "buzzer": os.environ.get("PARKGUARD_ESP32_BUZZER_PATH", "buzzer"),
    "lockdown": os.environ.get("PARKGUARD_ESP32_LOCKDOWN_PATH", "lockdown"),
    "unlock": os.environ.get("PARKGUARD_ESP32_UNLOCK_PATH", "unlock"),
    "greeting": os.environ.get("PARKGUARD_ESP32_GREETING_PATH", "greet"),
    "servo": os.environ.get("PARKGUARD_ESP32_SERVO_PATH", "servo"),
}

# Servo/greeting signalling is additive: it never changes the camera trigger or
# the security buzzer/lockdown flow.
ENABLE_GREETING_SIGNALS = os.environ.get(
    "PARKGUARD_GREETING_SIGNALS", "1"
).strip().lower() not in {"0", "false", "no", "off"}
SERVO_OPEN_ANGLE = max(0, min(180, _env_int("PARKGUARD_SERVO_OPEN_ANGLE", 90)))

# Local text-to-speech. pyttsx3 is preferred; on Windows a built-in SAPI
# fallback is used when pyttsx3 is not installed. Both run off the camera thread.
TTS_ENABLED = os.environ.get("PARKGUARD_TTS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TTS_RATE = max(80, min(300, _env_int("PARKGUARD_TTS_RATE", 170)))
TTS_VOLUME = min(1.0, max(0.0, _env_float("PARKGUARD_TTS_VOLUME", 1.0)))
TTS_TIMEOUT_SECONDS = max(2.0, _env_float("PARKGUARD_TTS_TIMEOUT_SECONDS", 15.0))

# Human-readable display names are stored in the permits table. The dataset
# folder name is only a fallback used when no display name has been enrolled.
ENABLE_PREVIEW = os.environ.get("PARKGUARD_PREVIEW", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DASHBOARD_HOST = os.environ.get("PARKGUARD_HOST", "0.0.0.0")
DASHBOARD_PORT = _env_int("PARKGUARD_PORT", 5000)


logger = logging.getLogger("parkguard")


# =============================================================================
# Domain models
# =============================================================================


@dataclass(frozen=True)
class PermitInfo:
    """A validated permit."""

    raw: str
    valid: bool
    normalized: Optional[str] = None
    is_resident: Optional[bool] = None
    floor: Optional[str] = None
    reason: str = ""
    student_id: Optional[str] = None
    display_name: Optional[str] = None


@dataclass(frozen=True)
class AuthorizationResult:
    """The result of matching one face against the enrolled database."""

    name: str
    authorized: bool
    distance: Optional[float]
    permit: Optional[PermitInfo]
    reason: str

    @property
    def overlay_label(self) -> str:
        if not self.authorized:
            return "Unknown"
        if self.permit and self.permit.normalized:
            return f"{self.name} | {self.permit.normalized}"
        return self.name


@dataclass(frozen=True)
class Detection:
    """One detected face in a frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    result: AuthorizationResult
    # True only when the face is large, textured, and has a valid encoding.
    clear: bool = True


@dataclass(frozen=True)
class SecurityDecision:
    """The cumulative security outcome of one logged access event."""

    unauthorized_streak: int
    lockdown: bool
    buzzer: bool
    signals: tuple[str, ...]


@dataclass
class FaceTrack:
    """A lightweight, IoU-based track that owns per-face timing state."""

    track_id: int
    bbox: tuple[int, int, int, int]
    result: AuthorizationResult
    identity_key: tuple[Any, ...]
    identity_since: float
    last_seen: float
    created_at: float
    captured: bool = False
    # Seconds of continuous, clear visibility. This is the real stability
    # countdown: it pauses when the face is missing or unclear and never counts
    # time spent away from the camera.
    visible_seconds: float = 0.0
    clear_visible: bool = False

    def ready_for_capture(self, now: float) -> bool:
        """Return True when this clear face has been stable long enough to log."""
        if self.captured or not self.clear_visible:
            return False
        return self.visible_seconds >= STABILITY_SECONDS


def validate_permit(
    permit_number: Any,
    student_id: Any = None,
    display_name: Any = None,
) -> PermitInfo:
    """
    Validate the project's four-digit permit format.

    1xxx = resident / upper floor
    0xxx = visitor / ground floor
    """
    raw = "" if permit_number is None else str(permit_number).strip()
    student = "" if student_id is None else str(student_id).strip() or None
    readable = "" if display_name is None else str(display_name).strip() or None

    if re.fullmatch(r"\d{4}", raw) is None:
        return PermitInfo(
            raw=raw,
            valid=False,
            reason="Permit must be exactly 4 digits (1xxx or 0xxx).",
            student_id=student,
            display_name=readable,
        )

    if raw[0] == "1":
        return PermitInfo(
            raw=raw,
            valid=True,
            normalized=raw,
            is_resident=True,
            floor="Upper",
            reason="Valid resident permit (upper floor).",
            student_id=student,
            display_name=readable,
        )
    if raw[0] == "0":
        return PermitInfo(
            raw=raw,
            valid=True,
            normalized=raw,
            is_resident=False,
            floor="Ground",
            reason="Valid visitor permit (ground floor).",
            student_id=student,
            display_name=readable,
        )

    return PermitInfo(
        raw=raw,
        valid=False,
        reason="Permit must start with 1 (resident) or 0 (visitor).",
        student_id=student,
        display_name=readable,
    )


def normalize_name(value: Any) -> str:
    """Normalize owner names for case/whitespace-insensitive joins."""
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def humanize_owner(value: Any) -> str:
    """Turn a dataset folder key into a readable fallback display name."""
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() == "unknown":
        return "Unknown"
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", text) if part)


def display_name_for(value: Any, explicit: Optional[str] = None) -> str:
    """Prefer an explicit DB display name, then a humanized folder name."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    return humanize_owner(value)


def greeting_name_for(result: "AuthorizationResult") -> str:
    """First name used by the ESP32 greeting, e.g. 'Alice' or 'Bob'."""
    explicit = result.permit.display_name if result.permit else None
    readable = display_name_for(result.name, explicit)
    parts = [part for part in readable.split() if part]
    return parts[0] if parts else "Guest"


# =============================================================================
# Local text-to-speech
# =============================================================================

_tts_lock = threading.Lock()
_tts_engine: Optional[Any] = None


def _powershell_path() -> Optional[str]:
    """Return the local PowerShell executable used by the Windows SAPI fallback."""
    if os.name != "nt":
        return None
    return shutil.which("powershell") or shutil.which("pwsh")


def tts_available() -> bool:
    """True when a local TTS backend is configured and available."""
    if not TTS_ENABLED:
        return False
    return pyttsx3 is not None or _powershell_path() is not None


def _init_tts_engine() -> Optional[Any]:
    """Create the pyttsx3 engine once, lazily, and keep it reusable."""
    global _tts_engine
    if pyttsx3 is None:
        return None

    with _tts_lock:
        if _tts_engine is not None:
            return _tts_engine
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            _tts_engine = engine
            logger.info(
                "TTS ready via pyttsx3 (rate=%s, volume=%s).",
                TTS_RATE,
                TTS_VOLUME,
            )
        except Exception as exc:
            logger.warning("pyttsx3 could not start: %s", exc)
            _tts_engine = None
        return _tts_engine


def _speak_with_pyttsx3(text: str) -> bool:
    global _tts_engine
    engine = _init_tts_engine()
    if engine is None:
        return False

    with _tts_lock:
        try:
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as exc:
            logger.warning("pyttsx3 speech failed: %s", exc)
            try:
                engine.stop()
            except Exception:
                pass
            _tts_engine = None
            return False


def _speak_with_windows_sapi(text: str) -> bool:
    """Fallback for Windows machines without pyttsx3 (uses built-in SAPI)."""
    executable = _powershell_path()
    if executable is None:
        return False

    rate = max(-10, min(10, int(round((TTS_RATE - 170) / 15))))
    volume = int(round(TTS_VOLUME * 100))
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {rate}; $s.Volume = {volume}; "
        "$s.Speak($env:PARKGUARD_TTS_TEXT); $s.Dispose();"
    )
    env = os.environ.copy()
    env["PARKGUARD_TTS_TEXT"] = text

    try:
        completed = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=TTS_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            logger.warning(
                "Windows SAPI TTS failed (%s): %s",
                completed.returncode,
                (completed.stderr or "").strip(),
            )
            return False
        return True
    except Exception as exc:
        logger.warning("Windows SAPI TTS failed: %s", exc)
        return False


def speak_welcome(name: str, *, wait: bool = False) -> bool:
    """
    Speak "Welcome <name>" without blocking the camera/trigger pipeline.

    pyttsx3 is used when available; otherwise Windows' built-in SAPI voice is
    used. If no backend is available the call is a safe no-op.
    """
    if not TTS_ENABLED:
        return False

    clean_name = " ".join(str(name or "").split()) or "Guest"
    text = f"Welcome {clean_name}"
    if not tts_available():
        logger.debug("TTS unavailable; skipping '%s'.", text)
        return False

    def _run() -> None:
        if pyttsx3 is not None and _speak_with_pyttsx3(text):
            return
        _speak_with_windows_sapi(text)

    if wait:
        _run()
        return True

    logger.info("[TTS] %s", text)
    threading.Thread(target=_run, name="parkguard-tts", daemon=True).start()
    return True


# =============================================================================
# SQLite layer
# =============================================================================

_db_init_lock = threading.Lock()
_db_ready = False


def _raw_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    """Add a column to an existing table. Safe to call on every startup."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def init_database() -> None:
    """Create the local schema if needed. Safe to call repeatedly."""
    global _db_ready
    with _db_init_lock:
        if _db_ready:
            return

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(_raw_connection()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS permits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    car_owner TEXT NOT NULL,
                    permit_number TEXT NOT NULL,
                    student_id TEXT,
                    display_name TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_permits_owner
                    ON permits(car_owner);

                CREATE TABLE IF NOT EXISTS face_encodings (
                    encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    car_owner TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    source_image TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(car_owner, source_image)
                );

                CREATE INDEX IF NOT EXISTS idx_face_encodings_owner
                    ON face_encodings(car_owner);

                CREATE TABLE IF NOT EXISTS access_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    student_id TEXT,
                    display_name TEXT,
                    permit_number TEXT,
                    is_resident INTEGER,
                    floor TEXT,
                    status TEXT NOT NULL,
                    reason TEXT,
                    image_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_access_logs_time
                    ON access_logs(event_time);

                CREATE TABLE IF NOT EXISTS security_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    unauthorized_streak INTEGER NOT NULL DEFAULT 0,
                    lockdown INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                INSERT OR IGNORE INTO security_state
                    (id, unauthorized_streak, lockdown, updated_at)
                VALUES (1, 0, 0, CURRENT_TIMESTAMP);
                """
            )
            # Add columns introduced by newer versions to older databases.
            _ensure_column(conn, "permits", "student_id", "TEXT")
            _ensure_column(conn, "permits", "display_name", "TEXT")
            _ensure_column(conn, "access_logs", "student_id", "TEXT")
            _ensure_column(conn, "access_logs", "display_name", "TEXT")
            conn.commit()

        _db_ready = True
        logger.info("SQLite ready: %s", DB_PATH)


def get_connection() -> sqlite3.Connection:
    """Return a new connection, initializing the schema on first use."""
    if not _db_ready:
        init_database()
    return _raw_connection()


def load_permit_map() -> dict[str, PermitInfo]:
    """
    Load permits keyed by normalized owner name.

    If an owner has both an invalid legacy permit and a valid 4-digit permit,
    the valid one wins.
    """
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT car_owner, permit_number, student_id, display_name FROM permits"
        ).fetchall()

    permits: dict[str, PermitInfo] = {}
    for row in rows:
        key = normalize_name(row["car_owner"])
        if not key:
            continue
        info = validate_permit(
            row["permit_number"],
            row["student_id"],
            row["display_name"] or display_name_for(row["car_owner"]),
        )
        current = permits.get(key)
        if current is None or (not current.valid and info.valid):
            permits[key] = info
    return permits


def store_face_encoding(car_owner: str, encoding: Any, source_image: str) -> bool:
    """Persist a 128-d face encoding. Returns True when a new row was inserted."""
    owner = str(car_owner).strip()
    if not owner:
        raise ValueError("car_owner must not be empty")

    array = np.asarray(encoding, dtype=np.float64).reshape(-1)
    if array.size != 128:
        raise ValueError(f"Expected a 128-d face encoding, got {array.size} values")

    source = str(source_image).strip() or f"{owner}#unknown"
    created_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO face_encodings
                (car_owner, encoding, source_image, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (owner, sqlite3.Binary(array.tobytes()), source, created_at),
        )
        conn.commit()
        return cursor.rowcount > 0


def load_face_encodings() -> tuple[np.ndarray, list[str]]:
    """Load every enrolled encoding and its owner from SQLite."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT car_owner, encoding FROM face_encodings ORDER BY encoding_id"
        ).fetchall()

    encodings: list[np.ndarray] = []
    names: list[str] = []
    for row in rows:
        blob = row["encoding"]
        if blob is None:
            continue
        array = np.frombuffer(blob, dtype=np.float64)
        if array.size != 128:
            logger.warning("Skipping malformed encoding for %s (size=%s)", row["car_owner"], array.size)
            continue
        encodings.append(array)
        names.append(str(row["car_owner"]))

    if not encodings:
        return np.empty((0, 128), dtype=np.float64), []

    return np.vstack(encodings).astype(np.float64, copy=False), names


def _enroll_from_pickle() -> int:
    if not ENCODINGS_PICKLE.is_file():
        return 0

    imported = 0
    try:
        with open(ENCODINGS_PICKLE, "rb") as handle:
            data = pickle.load(handle)
        names = data.get("names", []) if isinstance(data, dict) else []
        encodings = data.get("encodings", []) if isinstance(data, dict) else []
        for index, (name, encoding) in enumerate(zip(names, encodings)):
            try:
                if store_face_encoding(str(name), encoding, f"pickle#{index}"):
                    imported += 1
            except (TypeError, ValueError, sqlite3.Error) as exc:
                logger.warning("Could not import pickle encoding %s: %s", index, exc)
    except Exception as exc:  # malformed pickles must not prevent dataset enrollment
        logger.warning("Could not read %s: %s", ENCODINGS_PICKLE, exc)
    return imported


def _largest_face_location(locations: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
    if not locations:
        return None

    def area(location: tuple[int, int, int, int]) -> int:
        top, right, bottom, left = location
        return max(0, bottom - top) * max(0, right - left)

    return max(locations, key=area)


def _enroll_from_dataset() -> int:
    if not DATASET_DIR.is_dir():
        return 0

    imported = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    person_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    if person_dirs:
        logger.info("Enrolling face encodings from %s ...", DATASET_DIR)

    for person_dir in person_dirs:
        image_paths = sorted(
            path for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        for image_path in image_paths:
            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(image)
                largest = _largest_face_location(locations)
                if largest is None:
                    logger.warning("No face found in %s", image_path.name)
                    continue

                encodings = face_recognition.face_encodings(image, [largest])
                if not encodings:
                    logger.warning("No encoding produced for %s", image_path.name)
                    continue

                source = f"{image_path.name}#face0"
                if store_face_encoding(person_dir.name, encodings[0], source):
                    imported += 1
                else:
                    logger.debug("Encoding already enrolled: %s / %s", person_dir.name, source)
            except Exception as exc:  # dlib can raise various runtime errors for bad images
                logger.warning("Could not enroll %s: %s", image_path, exc)

    return imported


def bootstrap_face_encodings(force: bool = False) -> int:
    """
    Ensure SQLite has face encodings.

    When force is True the face_encodings table is rebuilt from
    encodings.pickle or dataset_faces. When False, existing rows are kept.
    """
    if force:
        with closing(get_connection()) as conn:
            conn.execute("DELETE FROM face_encodings")
            conn.commit()
    else:
        existing, _ = load_face_encodings()
        if existing.shape[0] > 0:
            return int(existing.shape[0])

    imported = _enroll_from_pickle()
    if imported == 0:
        imported = _enroll_from_dataset()

    if imported == 0:
        logger.warning(
            "No face encodings were enrolled. Add images under %s or create %s.",
            DATASET_DIR,
            ENCODINGS_PICKLE,
        )
    else:
        logger.info("Enrolled %d face encoding(s) into SQLite.", imported)
    return imported


def rebuild_face_encodings() -> int:
    """Delete and re-enroll all face encodings from local sources."""
    return bootstrap_face_encodings(force=True)


def record_access_event(
    *,
    event_time: str,
    person_name: str,
    permit_number: Optional[str],
    is_resident: Optional[bool],
    floor: Optional[str],
    status: str,
    reason: str,
    image_path: Optional[str],
    student_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    """Write one audit row to SQLite."""
    resident_value: Optional[int] = None
    if is_resident is not None:
        resident_value = 1 if is_resident else 0

    with closing(get_connection()) as conn:
        conn.execute(
            """
            INSERT INTO access_logs
                (event_time, person_name, student_id, display_name,
                 permit_number, is_resident, floor, status, reason, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_time,
                person_name,
                student_id,
                display_name,
                permit_number,
                resident_value,
                floor,
                status,
                reason,
                image_path,
            ),
        )
        conn.commit()


def _format_event_time(value: Any) -> str:
    try:
        parsed = dt.datetime.fromisoformat(str(value))
        return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (TypeError, ValueError):
        return str(value or "")


_CAPTURE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_CAPTURE_FILENAME_RE = re.compile(
    r"^capture_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>\d{2}-\d{2}-\d{2})-(?P<micro>\d{6})_"
    r"(?P<person>.+?)_"
    r"(?P<status>Authorized|Unauthorized)"
    r"_t(?P<track>\d+)\.(?P<ext>jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)


def scan_capture_files() -> dict[str, Path]:
    """Return the capture images currently present on disk, keyed by filename."""
    files: dict[str, Path] = {}
    try:
        if not CAPTURE_DIR.is_dir():
            return files
        for path in CAPTURE_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in _CAPTURE_IMAGE_SUFFIXES:
                files[path.name] = path
    except OSError as exc:
        logger.warning("Could not scan capture folder %s: %s", CAPTURE_DIR, exc)
    return files


def _event_from_capture_file(filename: str) -> Optional[dict[str, Any]]:
    """Build a dashboard event for an image that has no database row."""
    match = _CAPTURE_FILENAME_RE.match(filename)
    if match is None:
        return None

    raw_time = (
        f"{match.group('date')} "
        f"{match.group('time').replace('-', ':')}.{match.group('micro')}"
    )
    try:
        parsed = dt.datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S.%f").astimezone()
        event_time = parsed.isoformat(timespec="milliseconds")
    except ValueError:
        event_time = raw_time

    status = (
        "Authorized"
        if match.group("status").casefold() == "authorized"
        else "Unauthorized"
    )
    raw_person = match.group("person") or "Unknown"
    person_name = raw_person if status == "Authorized" else "Unknown"
    return {
        "log_id": None,
        "event_time": event_time,
        "person_name": person_name,
        "student_id": None,
        "display_name": display_name_for(person_name),
        "permit_number": None,
        "is_resident": None,
        "floor": None,
        "status": status,
        "reason": "Recovered from the capture folder (database row was missing).",
        "image_path": filename,
    }


def _event_sort_value(event: dict[str, Any]) -> float:
    parsed = _parse_event_datetime(event.get("event_time"))
    if parsed is None:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


def reconcile_events_with_disk(
    rows: Iterable[sqlite3.Row],
    disk_files: dict[str, Path],
) -> list[dict[str, Any]]:
    """
    Merge database rows with the images actually present in the capture folder.

    * Database rows whose image no longer exists are dropped (no ghost cards).
    * Images with no database row are recovered from the filename (no missing
      events after a database reset or an interrupted write).
    * Rows without an image path (manual/legacy records) are always kept.
    """
    events: list[dict[str, Any]] = []
    seen_images: set[str] = set()

    for row in rows:
        event = dict(row)
        image_path = event.get("image_path")
        if image_path:
            filename = os.path.basename(str(image_path))
            if filename not in disk_files:
                continue
            event["image_path"] = filename
            seen_images.add(filename)
        else:
            event["image_path"] = None
        events.append(event)

    for filename in sorted(disk_files):
        if filename in seen_images:
            continue
        recovered = _event_from_capture_file(filename)
        if recovered is not None:
            events.append(recovered)

    events.sort(key=_event_sort_value, reverse=True)
    return events


def fetch_recent_events(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent audit events, reconciled with the capture folder."""
    safe_limit = max(1, min(int(limit), 1000))
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT a.log_id, a.event_time, a.person_name,
                   COALESCE(
                       a.student_id,
                       (SELECT p.student_id FROM permits p
                        WHERE lower(p.car_owner) = lower(a.person_name)
                          AND p.student_id IS NOT NULL
                        ORDER BY p.id DESC LIMIT 1)
                   ) AS student_id,
                   COALESCE(
                       a.display_name,
                       (SELECT p.display_name FROM permits p
                        WHERE lower(p.car_owner) = lower(a.person_name)
                          AND p.display_name IS NOT NULL
                        ORDER BY p.id DESC LIMIT 1)
                   ) AS display_name,
                   a.permit_number, a.is_resident, a.floor, a.status,
                   a.reason, a.image_path
            FROM access_logs a
            ORDER BY a.log_id DESC
            """
        ).fetchall()

    events = reconcile_events_with_disk(rows, scan_capture_files())[:safe_limit]
    permit_map: Optional[dict[str, PermitInfo]] = None

    for event in events:
        image_path = event.get("image_path")
        event["image_filename"] = os.path.basename(image_path) if image_path else None
        event["time"] = _format_event_time(event.get("event_time"))

        # Recovered (orphan) authorized images can still show permit details by
        # looking up the canonical owner parsed from the filename.
        if event.get("log_id") is None and event.get("status") == "Authorized":
            if permit_map is None:
                try:
                    permit_map = load_permit_map()
                except sqlite3.Error:
                    permit_map = {}
            permit = permit_map.get(normalize_name(event.get("person_name")))
            if permit is not None and permit.valid:
                event["permit_number"] = permit.normalized
                event["is_resident"] = permit.is_resident
                event["floor"] = permit.floor
                event["student_id"] = permit.student_id
                event["display_name"] = display_name_for(
                    event.get("person_name"), permit.display_name
                )

        event["display_name"] = display_name_for(
            event.get("person_name"), event.get("display_name")
        )

        resident = event.get("is_resident")
        if resident is None:
            event["category"] = "—"
        elif int(resident) == 1:
            event["category"] = "ساكن / طابق علوي"
        else:
            event["category"] = "زائر / طابق أرضي"

        event["status_ar"] = (
            "مسموح بالدخول" if event.get("status") == "Authorized" else "غير مصرح"
        )

        parsed = _parse_event_datetime(event.get("event_time"))
        if parsed is not None:
            event["day_key"] = parsed.strftime("%Y-%m-%d")
            event["weekday_en"] = parsed.strftime("%A")
            event["weekday_ar"] = _WEEKDAYS_AR.get(parsed.weekday(), "")
            event["date_label"] = parsed.strftime("%Y-%m-%d")
            event["time_only"] = parsed.strftime("%I:%M:%S %p")
        else:
            event["day_key"] = "unknown"
            event["weekday_en"] = "Unknown"
            event["weekday_ar"] = "غير معروف"
            event["date_label"] = "—"
            event["time_only"] = event["time"]

    return events


_WEEKDAYS_AR = {
    0: "الاثنين",
    1: "الثلاثاء",
    2: "الأربعاء",
    3: "الخميس",
    4: "الجمعة",
    5: "السبت",
    6: "الأحد",
}


def _parse_event_datetime(value: Any) -> Optional[dt.datetime]:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed


def group_events_by_day(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group newest-first events into chronological day sections for the UI."""
    groups: list[dict[str, Any]] = []
    current_key: Optional[str] = None
    for event in events:
        key = str(event.get("day_key") or "unknown")
        if key != current_key:
            groups.append(
                {
                    "key": key,
                    "weekday_en": event.get("weekday_en", "Unknown"),
                    "weekday_ar": event.get("weekday_ar", "غير معروف"),
                    "date_label": event.get("date_label", "—"),
                    "events": [],
                    "total": 0,
                    "authorized": 0,
                    "unauthorized": 0,
                }
            )
            current_key = key

        group = groups[-1]
        group["events"].append(event)
        group["total"] += 1
        if event.get("status") == "Authorized":
            group["authorized"] += 1
        else:
            group["unauthorized"] += 1
    return groups


def build_week_summary(groups: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return aggregate totals for the visible day groups."""
    summary = {"days": 0, "total": 0, "authorized": 0, "unauthorized": 0}
    for group in groups:
        summary["days"] += 1
        summary["total"] += int(group.get("total", 0))
        summary["authorized"] += int(group.get("authorized", 0))
        summary["unauthorized"] += int(group.get("unauthorized", 0))
    return summary


# =============================================================================
# Cumulative security state (unauthorized streak / buzzer / lockdown)
# =============================================================================

_security_lock = threading.Lock()


def load_security_state() -> dict[str, Any]:
    """Read the persisted security counters."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT unauthorized_streak, lockdown, updated_at
            FROM security_state
            WHERE id = 1
            """
        ).fetchone()

    if row is None:
        return {"unauthorized_streak": 0, "lockdown": False, "updated_at": ""}

    return {
        "unauthorized_streak": int(row["unauthorized_streak"]),
        "lockdown": bool(row["lockdown"]),
        "updated_at": str(row["updated_at"]),
    }


def save_security_state(unauthorized_streak: int, lockdown: bool) -> None:
    """Persist the cumulative security counters."""
    updated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with closing(get_connection()) as conn:
        conn.execute(
            """
            INSERT INTO security_state
                (id, unauthorized_streak, lockdown, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                unauthorized_streak = excluded.unauthorized_streak,
                lockdown = excluded.lockdown,
                updated_at = excluded.updated_at
            """,
            (max(0, int(unauthorized_streak)), 1 if lockdown else 0, updated_at),
        )
        conn.commit()


def reset_security_state() -> dict[str, Any]:
    """Clear the unauthorized streak and any active lockdown."""
    with _security_lock:
        save_security_state(0, False)
        state = load_security_state()

    update_state(unauthorized_streak=0, lockdown=False)
    logger.info("Security state reset: streak=0, lockdown=off.")
    return state


def process_access_event(
    result: AuthorizationResult,
    *,
    send_signals: bool = True,
) -> SecurityDecision:
    """
    Apply one access event to the cumulative security state.

    The first unauthorized attempt only logs. From BUZZER_AFTER_ATTEMPTS onward
    the ESP32 buzzer sounds and lockdown is set. A successful authorized scan
    clears both the streak and the lockdown.
    """
    with _security_lock:
        current = load_security_state()
        prior_lockdown = bool(current["lockdown"])
        prior_streak = int(current["unauthorized_streak"])

        if result.authorized:
            streak = 0
            lockdown = False
            buzzer = False
            signals: list[str] = []
            if prior_lockdown:
                signals.append("unlock")
            signals.append("authorized")
        else:
            streak = prior_streak + 1
            buzzer = streak >= BUZZER_AFTER_ATTEMPTS
            lockdown = prior_lockdown or buzzer
            signals = ["unauthorized"]
            if buzzer:
                signals.append("buzzer")
                if not prior_lockdown:
                    signals.append("lockdown")

        save_security_state(streak, lockdown)

    update_state(unauthorized_streak=streak, lockdown=lockdown)
    logger.info(
        "Security: %s | streak=%d | buzzer=%s | lockdown=%s",
        "authorized" if result.authorized else "unauthorized",
        streak,
        "on" if buzzer else "off",
        "on" if lockdown else "off",
    )

    if send_signals:
        for endpoint in signals:
            send_esp32_signal(endpoint)

    return SecurityDecision(
        unauthorized_streak=streak,
        lockdown=lockdown,
        buzzer=buzzer,
        signals=tuple(signals),
    )


_state_lock = threading.Lock()
_system_state: dict[str, Any] = {
    "mode": "standby",
    "scan_pending": False,
    "camera_connected": False,
    "camera_index": CAMERA_INDEX,
    "camera_fps": 0.0,
    "known_faces": 0,
    "permits": 0,
    "unauthorized_streak": 0,
    "lockdown": False,
    "last_event": None,
    "last_frame_at": None,
    "last_scan_at": None,
    "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "version": "2.1",
}


def update_state(**kwargs: Any) -> None:
    with _state_lock:
        _system_state.update(kwargs)


def get_state() -> dict[str, Any]:
    with _state_lock:
        return dict(_system_state)


# =============================================================================
# Face recognition and tracking
# =============================================================================


def _face_distance(known_encodings: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Euclidean distance from one probe to every known encoding.

    This mirrors ``face_recognition.face_distance()`` without loading the dlib
    models, so permit/identity tests and database-only operations stay fast.
    """
    known = np.asarray(known_encodings, dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64).reshape(-1)
    if known.size == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(known - probe, axis=1)


class FaceRecognizer:
    """Loads encodings + permits from SQLite and matches live face encodings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._encodings = np.empty((0, 128), dtype=np.float64)
        self._names: list[str] = []
        self._permits: dict[str, PermitInfo] = {}
        self._last_refresh = 0.0

    @property
    def known_count(self) -> int:
        with self._lock:
            return len(self._names)

    @property
    def permit_count(self) -> int:
        with self._lock:
            return len(self._permits)

    def refresh(self, force: bool = False) -> None:
        """Reload encodings and permits from SQLite at most every N seconds."""
        now = time.monotonic()
        with self._refresh_lock:
            if not force and (now - self._last_refresh) < DB_REFRESH_SECONDS:
                return

            try:
                encodings, names = load_face_encodings()
                permits = load_permit_map()
            except sqlite3.Error as exc:
                logger.error("Could not refresh face database: %s", exc)
                return

            with self._lock:
                self._encodings = encodings
                self._names = names
                self._permits = permits
            self._last_refresh = time.monotonic()
            update_state(known_faces=len(names), permits=len(permits))

    def initialize(self, force_rebuild: bool = False) -> None:
        """Bootstrap the DB and load the initial identity/permit snapshot."""
        init_database()
        bootstrap_face_encodings(force=force_rebuild)
        self.refresh(force=True)

        with self._lock:
            permits = dict(self._permits)
            encodings = self._encodings.shape[0]

        for owner, permit in sorted(permits.items()):
            if not permit.valid:
                logger.warning(
                    "Invalid permit for '%s': %s Update permits.permit_number "
                    "to a 4-digit value such as 1001 (resident) or 0456 (visitor).",
                    owner,
                    permit.reason,
                )
        logger.info(
            "Identity database loaded: %d encoding(s), %d permit(s).",
            encodings,
            len(permits),
        )

    def identify(self, encoding: Any) -> AuthorizationResult:
        """Match one face encoding and verify the matched owner's permit."""
        self.refresh()

        with self._lock:
            encodings = self._encodings
            names = list(self._names)
            permits = dict(self._permits)

        if encodings.shape[0] == 0:
            return AuthorizationResult("Unknown", False, None, None, "No enrolled faces.")

        probe = np.asarray(encoding, dtype=np.float64).reshape(-1)
        if probe.size != 128:
            return AuthorizationResult("Unknown", False, None, None, "Invalid face encoding.")

        distances = _face_distance(encodings, probe)
        best_index = int(np.argmin(distances))
        distance = float(distances[best_index])

        if (not np.isfinite(distance)) or distance > FACE_TOLERANCE:
            return AuthorizationResult("Unknown", False, distance, None, "Unknown face.")

        name = names[best_index]
        permit = permits.get(normalize_name(name))
        if permit is None:
            return AuthorizationResult(
                name,
                False,
                distance,
                None,
                f"Recognized as {name}, but no permit on file.",
            )
        if not permit.valid:
            return AuthorizationResult(
                name,
                False,
                distance,
                permit,
                f"Recognized as {name}, but {permit.reason}",
            )

        return AuthorizationResult(name, True, distance, permit, "Authorized.")


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_ratio(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    first_center = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    second_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    size = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1)
    return distance / size


def _identity_key(result: AuthorizationResult) -> tuple[Any, ...]:
    permit_number = result.permit.normalized if result.permit else None
    return (result.name, result.authorized, permit_number, result.reason)


class FaceTracker:
    """Small IoU tracker that keeps hold/capture state per visible face."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: list[FaceTrack] = []
        self._next_id = 1
        self._last_update_at: Optional[float] = None

    @property
    def tracks(self) -> list[FaceTrack]:
        with self._lock:
            return list(self._tracks)

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._last_update_at = None

    def update(self, detections: Iterable[Detection], now: float) -> list[FaceTrack]:
        with self._lock:
            previous_update_at = self._last_update_at
            matched_ids: set[int] = set()

            for detection in detections:
                candidates: list[tuple[float, FaceTrack]] = []
                for track in self._tracks:
                    if track.track_id in matched_ids:
                        continue
                    iou = _iou(track.bbox, detection.bbox)
                    center_ratio = _center_ratio(track.bbox, detection.bbox)
                    if iou >= IOU_MATCH_THRESHOLD:
                        candidates.append((2.0 + iou, track))
                    elif center_ratio <= CENTER_MATCH_RATIO:
                        candidates.append((1.0 - center_ratio, track))

                if candidates:
                    _, best_track = max(candidates, key=lambda item: item[0])
                    matched_ids.add(best_track.track_id)
                    self._apply(best_track, detection, now, previous_update_at)
                    continue

                track = FaceTrack(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    result=detection.result,
                    identity_key=_identity_key(detection.result),
                    identity_since=now,
                    last_seen=now,
                    created_at=now,
                    clear_visible=detection.clear,
                )
                self._next_id += 1
                self._tracks.append(track)

            self._tracks = [
                track for track in self._tracks
                if (now - track.last_seen) <= TRACK_TTL_SECONDS
            ]
            self._last_update_at = now
            return list(self._tracks)

    @staticmethod
    def _apply(
        track: FaceTrack,
        detection: Detection,
        now: float,
        previous_update_at: Optional[float],
    ) -> None:
        new_identity = _identity_key(detection.result)
        identity_changed = new_identity != track.identity_key
        if identity_changed:
            # A different person/result now occupies this track: restart timers.
            track.identity_key = new_identity
            track.identity_since = now
            track.visible_seconds = 0.0
            track.captured = False

        # Accrue stability time only for a clear face that was also present in
        # the previous processed frame. A missing frame, a long stall, or an
        # unclear detection pauses the countdown instead of consuming it.
        continuous = (
            previous_update_at is not None
            and track.last_seen == previous_update_at
            and (now - track.last_seen) <= STABILITY_FRAME_GAP_SECONDS
        )
        elapsed = 0.0
        if continuous and not identity_changed:
            elapsed = max(0.0, now - track.last_seen)
        if detection.clear and elapsed > 0:
            track.visible_seconds += elapsed

        track.bbox = detection.bbox
        track.result = detection.result
        track.last_seen = now
        track.clear_visible = detection.clear


# =============================================================================
# Drawing
# =============================================================================


def _clamp_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width - 1))
    y2 = max(y1 + 1, min(int(y2), height - 1))
    return x1, y1, x2, y2


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a filled label immediately above the box (or below if near the top)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    padding = 6

    bottom = y - 8
    top = bottom - text_height - baseline - padding
    if top < 0:
        bottom = y + text_height + baseline + padding + 8
        top = bottom - text_height - baseline - padding

    left = max(0, min(int(x), max(0, frame.shape[1] - text_width - (2 * padding) - 1)))
    right = min(frame.shape[1] - 1, left + text_width + (2 * padding))
    top = max(0, top)
    bottom = max(top + text_height + baseline + padding, min(frame.shape[0] - 1, bottom + padding))

    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (left + padding, max(top + text_height + 2, bottom - padding)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_reason(frame: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    line_y = min(frame.shape[0] - 8, y + 24)
    cv2.putText(frame, text, (x, line_y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, line_y), font, scale, (0, 0, 255), 1, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray, tracks: Iterable[FaceTrack]) -> np.ndarray:
    """Draw all boxes/labels in place and return the frame."""
    for track in tracks:
        result = track.result
        x1, y1, x2, y2 = _clamp_bbox(frame, track.bbox)

        if result.authorized:
            color = (0, 255, 0)  # green
            thickness = 2
            label = result.overlay_label
        else:
            color = (0, 0, 255)  # red
            thickness = 3
            label = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_label(frame, label, x1, y1, color)

        # Keep the overlay exactly "Unknown" above the box, but show a short
        # diagnostics line below it when a known person has a bad permit.
        if not result.authorized and result.reason not in {"Unknown face.", "Unknown face"}:
            _draw_reason(frame, result.reason, x1, y2)

    return frame


def draw_hud(frame: np.ndarray, fps: float, known_count: int, permit_count: int) -> None:
    """Draw a compact status bar without obscuring the face labels."""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 34), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status = f"ParkGuard LIVE | {fps:4.1f} FPS | Known: {known_count} | Permits: {permit_count}"
    cv2.putText(frame, status, (10, 23), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (text_width, _), _ = cv2.getTextSize(timestamp, font, 0.55, 1)
    cv2.putText(
        frame,
        timestamp,
        (max(10, width - text_width - 10), 23),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# =============================================================================
# Capture + ESP32 signalling
# =============================================================================


def _safe_filename_component(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.")
    return text[:48] or "person"


def send_esp32_signal(endpoint: str, params: Optional[dict[str, Any]] = None) -> None:
    """Send a non-blocking local HTTP signal to the ESP32 gate controller."""
    host = ESP32_IP.replace("http://", "").replace("https://", "").strip("/")
    if not host:
        return

    path = ESP32_ENDPOINTS.get(endpoint, endpoint)
    url = f"http://{host}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"

    def _send() -> None:
        try:
            response = requests.get(url, timeout=ESP32_TIMEOUT_SECONDS)
            logger.info("ESP32 %s -> HTTP %s", url, response.status_code)
        except requests.RequestException as exc:
            logger.warning("ESP32 %s failed: %s", url, exc)

    threading.Thread(target=_send, name=f"esp32-{endpoint}", daemon=True).start()


def announce_authorized(result: AuthorizationResult) -> Optional[str]:
    """
    Greet an authorized user and open the servo.

    This runs after the existing security signals. It is intentionally additive
    so the camera trigger, recognition, and buzzer/lockdown paths stay untouched.
    """
    if not result.authorized:
        return None

    name = greeting_name_for(result)
    message = f"Welcome {name}"
    logger.info("[GREETING] %s | user=%s", message, result.name)

    # Local audio greeting (pyttsx3 or Windows SAPI) runs in the background.
    speak_welcome(name)

    if not ENABLE_GREETING_SIGNALS:
        return message

    send_esp32_signal(
        "greeting",
        params={"name": name, "message": message},
    )
    send_esp32_signal(
        "servo",
        params={"action": "open", "angle": SERVO_OPEN_ANGLE},
    )

    # Serial is optional and silently disabled when pyserial or a port is absent.
    esp32_comm.send_command(f"WELCOME {name}")
    esp32_comm.send_command(f"SERVO OPEN {SERVO_OPEN_ANGLE}")
    return message


def save_capture(frame: np.ndarray, track: FaceTrack) -> Optional[str]:
    """
    Save the annotated frame with a precise timestamp and write an audit row.

    Returns the filename on success, or None if the image could not be written.
    A track is captured at most once until its identity changes or it leaves.
    """
    if track.captured:
        return None

    result = track.result
    now = dt.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    status = "Authorized" if result.authorized else "Unauthorized"
    recognition_name = result.name if result.authorized else "Unknown"
    filename = (
        f"capture_{timestamp}_{_safe_filename_component(recognition_name)}"
        f"_{status}_t{track.track_id}.jpg"
    )

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = CAPTURE_DIR / filename
    try:
        written = cv2.imwrite(
            str(image_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
    except cv2.error as exc:
        logger.error("Could not write capture %s: %s", image_path, exc)
        return None

    if not written:
        logger.error("OpenCV refused to write capture %s", image_path)
        return None

    permit = result.permit
    permit_valid = bool(result.authorized and permit and permit.valid)
    permit_number = permit.raw if permit_valid and permit else None
    is_resident = permit.is_resident if permit_valid and permit else None
    floor = permit.floor if permit_valid and permit else None
    student_id = permit.student_id if permit_valid and permit else None
    # Rendering and logging must always agree: only an authorized match shows
    # and stores the real name. Everything else is stored as "Unknown".
    person_name = recognition_name
    readable_name = (
        display_name_for(recognition_name, permit.display_name)
        if permit_valid and permit
        else "Unknown"
    )
    event_time = now.astimezone().isoformat(timespec="milliseconds")

    try:
        record_access_event(
            event_time=event_time,
            person_name=person_name,
            permit_number=permit_number,
            is_resident=is_resident,
            floor=floor,
            status=status,
            reason=result.reason,
            image_path=filename,
            student_id=student_id,
            display_name=readable_name,
        )
    except sqlite3.Error as exc:
        logger.error("Could not write access log for %s: %s", filename, exc)

    track.captured = True
    update_state(
        last_event={
            "time": _format_event_time(event_time),
            "name": readable_name,
            "person_name": person_name,
            "student_id": student_id,
            "permit_number": permit_number,
            "status": status,
            "reason": result.reason,
            "image_filename": filename,
        }
    )

    process_access_event(result)
    announce_authorized(result)
    logger.info(
        "[CAPTURE] %s | %s | name=%s | student_id=%s | permit=%s",
        filename,
        status,
        readable_name,
        student_id or "-",
        permit_number or "-",
    )
    return filename


# =============================================================================
# Camera processing loop
# =============================================================================


def _face_is_clear(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    encoding: Any,
) -> bool:
    """
    Decide whether a detected face is clear enough to advance the countdown.

    A covered face usually produces no detection at all. When dlib still returns
    a box, reject crops that are too small or nearly flat/dark (a hand, mask, or
    obstructed lens). This keeps the stability countdown paused instead of
    counting frames where there is no usable face.
    """
    x1, y1, x2, y2 = _clamp_bbox(frame, bbox)
    if (x2 - x1) < MIN_CLEAR_FACE_PIXELS or (y2 - y1) < MIN_CLEAR_FACE_PIXELS:
        return False

    try:
        probe = np.asarray(encoding, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    if probe.size != 128 or not np.all(np.isfinite(probe)):
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return False
    if float(gray.std()) < MIN_CLEAR_FACE_CONTRAST:
        return False
    return True


def process_frame(
    frame: np.ndarray,
    recognizer: FaceRecognizer,
    tracker: FaceTracker,
    now: float,
) -> tuple[np.ndarray, list[FaceTrack]]:
    """
    Detect, identify, track, and annotate one frame.

    Returns the annotated frame and the tracks that are ready to be captured.
    """
    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    try:
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=1,
            model="hog",
        )
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    except Exception as exc:
        logger.warning("Face detection failed for this frame: %s", exc)
        return frame, []

    scale_x = width / small_frame.shape[1] if small_frame.shape[1] else 1.0
    scale_y = height / small_frame.shape[0] if small_frame.shape[0] else 1.0

    detections: list[Detection] = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        bbox = (
            int(round(left * scale_x)),
            int(round(top * scale_y)),
            int(round(right * scale_x)),
            int(round(bottom * scale_y)),
        )
        try:
            result = recognizer.identify(encoding)
        except Exception:
            logger.exception("Could not identify one face; treating it as Unknown for this frame.")
            result = AuthorizationResult("Unknown", False, None, None, "Unknown face.")
        detections.append(
            Detection(
                bbox=bbox,
                result=result,
                clear=_face_is_clear(frame, bbox, encoding),
            )
        )

    tracks = tracker.update(detections, now)
    annotate_frame(frame, tracks)

    # Only capture tracks that are clear and visible in this frame, and whose
    # stability countdown has actually completed while the face stayed visible.
    due_tracks = [
        track for track in tracks
        if track.last_seen == now and track.ready_for_capture(now)
    ]
    return frame, due_tracks


def _open_camera() -> Optional[cv2.VideoCapture]:
    try:
        capture = cv2.VideoCapture(CAMERA_INDEX)
    except cv2.error as exc:
        logger.error("Could not create camera capture: %s", exc)
        return None

    if not capture.isOpened():
        capture.release()
        return None

    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass
    return capture


# -----------------------------------------------------------------------------
# Trigger queue
# -----------------------------------------------------------------------------

_scan_lock = threading.Lock()
_scan_pending = threading.Event()


def request_scan() -> bool:
    """
    Queue exactly one scan session.

    Called by the ESP32 ultrasonic sensor through /start_camera or /trigger.
    Returns True when this request was newly queued, False when a request was
    already waiting (extra triggers coalesce into one session).
    """
    with _scan_lock:
        already_pending = _scan_pending.is_set()
        _scan_pending.set()

    update_state(scan_pending=True)
    logger.info(
        "ESP32 scan trigger queued%s.",
        " (already pending)" if already_pending else "",
    )
    return not already_pending


def consume_scan_request() -> bool:
    """Take the queued trigger, if any. Only one camera session may consume it."""
    with _scan_lock:
        had_request = _scan_pending.is_set()
        _scan_pending.clear()

    update_state(scan_pending=False)
    return had_request


def clear_scan_requests() -> None:
    """Drop any queued triggers (used on reset and after a failed session)."""
    with _scan_lock:
        _scan_pending.clear()
    update_state(scan_pending=False)


def has_pending_scan() -> bool:
    """Return True while a trigger is waiting for the camera session."""
    return _scan_pending.is_set()


def run_pending_scan_session(
    recognizer: FaceRecognizer,
    tracker: FaceTracker,
    stop_event: threading.Event,
    preview: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Run one ESP32-triggered scan session.

    Returns immediately without touching the camera when no trigger is queued.
    The camera is opened for this session only, one access event is captured,
    then the camera is released and the system returns to standby.
    """
    if not consume_scan_request():
        return None

    if preview is None:
        preview = ENABLE_PREVIEW
    if timeout is None:
        timeout = SCAN_SESSION_TIMEOUT_SECONDS

    window_name = "ParkGuard - Triggered Gate Scan"
    tracker.reset()
    update_state(
        mode="active",
        scan_pending=False,
        camera_connected=False,
        last_scan_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    logger.info("Ultrasonic trigger received; starting one scan session (%.1fs max).", timeout)

    capture: Optional[cv2.VideoCapture] = None
    filename: Optional[str] = None
    deadline = time.monotonic() + max(0.1, float(timeout))

    try:
        capture = _open_camera()
        if capture is None:
            update_state(camera_connected=False)
            logger.warning("Camera %s is unavailable; scan session skipped.", CAMERA_INDEX)
            return None

        update_state(camera_connected=True, camera_index=CAMERA_INDEX)
        while not stop_event.is_set() and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.warning("Camera frame read failed during scan session; ending session.")
                break

            now = time.monotonic()
            try:
                frame, due_tracks = process_frame(frame, recognizer, tracker, now)
            except Exception:
                logger.exception("Frame-processing error during scan session; continuing.")
                due_tracks = []

            draw_hud(frame, 0.0, recognizer.known_count, recognizer.permit_count)

            for track in due_tracks:
                try:
                    saved = save_capture(frame, track)
                except Exception:
                    logger.exception(
                        "Could not save capture for track %s during scan session.",
                        track.track_id,
                    )
                    continue
                if saved:
                    filename = saved
                    break

            if preview:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested from preview window.")
                        stop_event.set()
                        break
                except cv2.error:
                    logger.warning("Preview window is unavailable; continuing headless.")
                    preview = False

            if filename:
                break

            update_state(
                last_frame_at=dt.datetime.now().astimezone().isoformat(timespec="seconds")
            )
            time.sleep(0.001)
    finally:
        if capture is not None:
            capture.release()
        tracker.reset()
        update_state(
            camera_connected=False,
            mode="standby",
            scan_pending=has_pending_scan(),
        )
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        logger.info("Scan session finished: %s", filename or "no access event captured.")

    return filename


def run_trigger_service(
    recognizer: FaceRecognizer,
    tracker: FaceTracker,
    stop_event: threading.Event,
    preview: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> None:
    """
    Standby loop.

    The camera is never opened here. The service only wakes when the ESP32
    queues a trigger through /start_camera or /trigger, then runs exactly one
    scan session and returns to standby.
    """
    update_state(
        mode="standby",
        scan_pending=has_pending_scan(),
        camera_connected=False,
    )
    logger.info("ParkGuard standby: waiting for an ESP32 ultrasonic trigger.")

    while not stop_event.is_set():
        if not _scan_pending.wait(SCAN_IDLE_POLL_SECONDS):
            continue

        try:
            run_pending_scan_session(
                recognizer,
                tracker,
                stop_event,
                preview=preview,
                timeout=timeout,
            )
        except Exception:
            logger.exception("Scan session failed; returning to standby.")
        finally:
            # A session normally consumes the request itself. If it failed
            # before doing so, drop the stale request instead of busy-spinning.
            if _scan_pending.is_set():
                clear_scan_requests()
            update_state(mode="standby", camera_connected=False)

    logger.info("Trigger service stopped.")


# =============================================================================
# Flask dashboard
# =============================================================================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ParkGuard</title>
    <style>
        :root {
            --ink: #0f172a;
            --muted: #64748b;
            --line: #e2e8f0;
            --brand: #2563eb;
            --brand-dark: #1d4ed8;
            --ok: #059669;
            --ok-soft: #ecfdf5;
            --bad: #dc2626;
            --bad-soft: #fef2f2;
            --panel: #ffffff;
            --bg: #eef4ff;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            padding: 28px 18px 48px;
            color: var(--ink);
            font-family: "Segoe UI", Tahoma, Arial, sans-serif;
            background:
                radial-gradient(circle at 12% 8%, #dbeafe 0, transparent 34%),
                radial-gradient(circle at 88% 2%, #ede9fe 0, transparent 30%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
        }
        .page { width: min(1220px, 100%); margin: 0 auto; }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 28px 30px;
            border-radius: 24px;
            color: #fff;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
            box-shadow: 0 24px 60px rgba(30, 58, 138, 0.28);
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            top: -170px;
            left: -80px;
            background: rgba(255, 255, 255, 0.12);
        }
        .hero-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
        .brand { display: flex; align-items: center; gap: 14px; }
        .brand-mark {
            width: 54px; height: 54px; border-radius: 17px;
            display: grid; place-items: center; font-size: 27px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.22);
        }
        .hero h1 { margin: 0; font-size: clamp(22px, 3vw, 31px); letter-spacing: -0.4px; }
        .hero p { margin: 5px 0 0; color: #c7d7ff; font-size: 14px; }
        .live-pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 9px 14px; border-radius: 999px; font-size: 13px; font-weight: 700;
            background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.24);
        }
        .live-dot { width: 9px; height: 9px; border-radius: 50%; background: #34d399; box-shadow: 0 0 0 5px rgba(52, 211, 153, 0.18); }
        .live-dot.idle { background: #fbbf24; box-shadow: 0 0 0 5px rgba(251, 191, 36, 0.18); }
        .live-dot.alert { background: #f87171; box-shadow: 0 0 0 5px rgba(248, 113, 113, 0.18); }
        .stats {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 14px; margin-top: 22px;
        }
        .stat {
            padding: 16px 18px; border-radius: 18px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.16);
            backdrop-filter: blur(8px);
        }
        .stat-label { display: block; font-size: 12px; color: #c7d7ff; margin-bottom: 7px; }
        .stat-value { display: block; font-size: 19px; font-weight: 800; }
        .stat-value.ok { color: #6ee7b7; }
        .stat-value.bad { color: #fca5a5; }
        .toolbar {
            display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
            margin: 18px 0 26px;
        }
        .toolbar form { margin: 0; }
        .btn {
            border: 0; border-radius: 12px; padding: 11px 18px;
            font-size: 14px; font-weight: 700; cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn-primary { background: var(--brand); color: #fff; box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28); }
        .btn-primary:hover { background: var(--brand-dark); }
        .btn-ghost { background: #fff; color: #334155; border: 1px solid var(--line); }
        .toolbar-note { margin-inline-start: auto; color: var(--muted); font-size: 13px; }
        .section-head { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
        .section-head h2 { margin: 0; font-size: 22px; letter-spacing: -0.3px; }
        .section-head p { margin: 3px 0 0; color: var(--muted); font-size: 13px; }
        .week-summary {
            padding: 18px; margin-bottom: 22px;
            border-radius: 22px; background: #fff;
            border: 1px solid var(--line);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        }
        .week-summary-head {
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; flex-wrap: wrap; margin-bottom: 14px;
        }
        .week-title { font-size: 17px; font-weight: 800; letter-spacing: -0.2px; }
        .week-meta { margin-inline-start: 8px; color: var(--muted); font-size: 13px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .summary-card {
            display: block; padding: 13px 14px; border-radius: 15px;
            color: inherit; text-decoration: none; background: #f8fafc;
            border: 1px solid var(--line);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        .summary-card:hover { transform: translateY(-2px); border-color: #bfdbfe; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.1); }
        .summary-card.active {
            background: linear-gradient(135deg, #eff6ff, #e0e7ff);
            border-color: #93c5fd;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
        }
        .summary-day { font-size: 15px; font-weight: 800; }
        .summary-date { margin-top: 2px; color: var(--muted); font-size: 12px; direction: ltr; }
        .summary-total { margin-top: 9px; font-size: 13px; font-weight: 700; }
        .summary-counts { display: flex; gap: 9px; margin-top: 5px; font-size: 12px; font-weight: 700; }
        .summary-counts .ok { color: var(--ok); }
        .summary-counts .bad { color: var(--bad); }
        .summary-empty { color: var(--muted); font-size: 13px; padding: 8px 2px; }
        .btn-small { display: inline-block; padding: 8px 12px; font-size: 12px; text-decoration: none; }
        .btn-small.active { background: #eff6ff; color: var(--brand-dark); border-color: #93c5fd; }
        .day-group { margin-bottom: 30px; }
        .day-header {
            display: flex; align-items: center; justify-content: space-between; gap: 12px;
            padding: 13px 18px; margin-bottom: 14px;
            border-radius: 16px; background: #fff;
            border: 1px solid var(--line);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        .day-title { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
        .day-en { font-size: 17px; font-weight: 800; letter-spacing: -0.2px; }
        .day-ar { color: var(--muted); font-size: 14px; font-weight: 600; }
        .day-date { padding: 6px 12px; border-radius: 999px; background: #eff6ff; color: var(--brand-dark); font-size: 13px; font-weight: 700; direction: ltr; }
        .event-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
        .event-card {
            overflow: hidden; border-radius: 20px; background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .event-card:hover { transform: translateY(-3px); box-shadow: 0 20px 44px rgba(15, 23, 42, 0.12); }
        .event-card.authorized { border-top: 4px solid var(--ok); }
        .event-card.denied { border-top: 4px solid var(--bad); }
        .event-media { position: relative; height: 190px; background: #f1f5f9; }
        .event-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .no-image { height: 100%; display: grid; place-items: center; color: #94a3b8; font-size: 28px; }
        .badge {
            position: absolute; top: 12px; inset-inline-start: 12px;
            padding: 7px 12px; border-radius: 999px;
            font-size: 12px; font-weight: 800;
            backdrop-filter: blur(6px);
        }
        .badge-ok { color: #065f46; background: rgba(236, 253, 245, 0.94); }
        .badge-bad { color: #991b1b; background: rgba(254, 242, 242, 0.94); }
        .event-body { padding: 16px 17px 18px; }
        .event-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
        .event-heading h3 { margin: 0; font-size: 18px; letter-spacing: -0.2px; }
        .event-heading time { color: var(--muted); font-size: 12px; direction: ltr; white-space: nowrap; }
        .facts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
        .fact { padding: 10px 11px; border-radius: 12px; background: #f8fafc; border: 1px solid #eef2f7; }
        .fact span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }
        .fact strong { display: block; font-size: 14px; word-break: break-word; }
        .fact .mono { direction: ltr; text-align: right; font-variant-numeric: tabular-nums; }
        .reason {
            margin: 12px 0 0; padding: 10px 12px; border-radius: 12px;
            color: #991b1b; background: var(--bad-soft); font-size: 12px; line-height: 1.6;
        }
        .empty {
            padding: 54px 20px; text-align: center; border-radius: 22px;
            background: #fff; border: 1px dashed #cbd5e1; color: var(--muted);
        }
        .empty strong { display: block; color: var(--ink); font-size: 18px; margin-bottom: 6px; }
        @media (max-width: 560px) {
            body { padding: 14px 10px 32px; }
            .hero { padding: 22px 18px; border-radius: 20px; }
            .facts { grid-template-columns: 1fr; }
            .toolbar-note { width: 100%; margin: 0; }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <div class="hero-top">
                <div class="brand">
                    <div class="brand-mark">🛡️</div>
                    <div>
                        <h1>ParkGuard</h1>
                        <p>نظام التحقق من الوجوه وبوابة الدخول الذكية — يعمل محلياً بالكامل</p>
                    </div>
                </div>
                <div class="live-pill">
                    <span class="live-dot {% if state.mode == 'active' or state.scan_pending %}idle{% elif state.lockdown %}alert{% endif %}"></span>
                    {% if state.mode == 'active' %}
                        جلسة فحص جارية
                    {% elif state.scan_pending %}
                        إشارة مستشعر في الانتظار
                    {% else %}
                        وضع الاستعداد
                    {% endif %}
                </div>
            </div>
            <div class="stats">
                <div class="stat">
                    <span class="stat-label">حالة الكاميرا</span>
                    <span class="stat-value {% if state.mode == 'active' or state.scan_pending %}ok{% endif %}">
                        {% if state.mode == 'active' %}تعمل الآن{% elif state.scan_pending %}بانتظار الفحص{% else %}استعداد{% endif %}
                    </span>
                </div>
                <div class="stat">
                    <span class="stat-label">الحالة الأمنية</span>
                    <span class="stat-value {% if state.lockdown %}bad{% else %}ok{% endif %}">
                        {% if state.lockdown %}🔒 قفل مفعّل{% else %}آمن{% endif %}
                    </span>
                </div>
                <div class="stat">
                    <span class="stat-label">البصمات المسجلة</span>
                    <span class="stat-value">{{ state.known_faces }}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">التصاريح</span>
                    <span class="stat-value">{{ state.permits }}</span>
                </div>
            </div>
        </section>

        <div class="toolbar">
            <form method="post" action="{{ url_for('start_camera_route') }}">
                <button class="btn btn-primary" type="submit">تشغيل فحص الآن</button>
            </form>
            <form method="post" action="{{ url_for('reset_security_route') }}">
                <button class="btn btn-ghost" type="submit">إعادة تعيين الحالة الأمنية</button>
            </form>
            <span class="toolbar-note">
                المحاولات غير المصرح بها: {{ state.unauthorized_streak }} · التصاريح 1xxx ساكن/علوي، 0xxx زائر/أرضي
            </span>
        </div>

        <section>
            <div class="section-head">
                <div>
                    <h2>سجل محاولات الدخول</h2>
                    <p>
                        مجموعة زمنياً حسب اليوم —
                        {% if show_all %}
                            عرض كل الأيام
                        {% elif selected_group %}
                            {{ selected_group.weekday_en }} · {{ selected_group.date_label }}
                        {% else %}
                            الأحدث أولاً
                        {% endif %}
                    </p>
                </div>
            </div>

            <div class="week-summary">
                <div class="week-summary-head">
                    <div>
                        <span class="week-title">الملخص الأسبوعي</span>
                        <span class="week-meta">
                            {{ week_summary.days }} يوم · {{ week_summary.total }} حدث
                        </span>
                    </div>
                    <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
                        <span class="week-meta">
                            <span class="ok">✅ {{ week_summary.authorized }}</span>
                            ·
                            <span class="bad">⛔ {{ week_summary.unauthorized }}</span>
                        </span>
                        <a class="btn btn-ghost btn-small {% if show_all %}active{% endif %}"
                           href="{{ url_for('home', day='all') }}">عرض كل الأيام</a>
                    </div>
                </div>
                <div class="summary-grid">
                    {% for group in groups %}
                    <a class="summary-card {% if not show_all and group.key == selected_day %}active{% endif %}"
                       href="{{ url_for('home', day=group.key) }}">
                        <div class="summary-day">
                            {{ group.weekday_en }}
                            <span class="day-ar">{{ group.weekday_ar }}</span>
                        </div>
                        <div class="summary-date">{{ group.date_label }}</div>
                        <div class="summary-total">{{ group.total }} حدث</div>
                        <div class="summary-counts">
                            <span class="ok">✅ {{ group.authorized }}</span>
                            <span class="bad">⛔ {{ group.unauthorized }}</span>
                        </div>
                    </a>
                    {% else %}
                    <div class="summary-empty">لا توجد أيام لعرضها بعد.</div>
                    {% endfor %}
                </div>
            </div>

            {% for group in visible_groups %}
            <section class="day-group">
                <header class="day-header">
                    <div class="day-title">
                        <span class="day-en">{{ group.weekday_en }} Logs</span>
                        <span class="day-ar">سجلات {{ group.weekday_ar }}</span>
                    </div>
                    <span class="day-date">{{ group.date_label }}</span>
                </header>
                <div class="event-grid">
                    {% for capture in group.events %}
                    <article class="event-card {% if capture.status == 'Authorized' %}authorized{% else %}denied{% endif %}">
                        <div class="event-media">
                            {% if capture.image_filename %}
                                <img src="{{ url_for('get_capture_image', filename=capture.image_filename) }}" alt="capture">
                            {% else %}
                                <div class="no-image">—</div>
                            {% endif %}
                            <span class="badge {% if capture.status == 'Authorized' %}badge-ok{% else %}badge-bad{% endif %}">
                                {% if capture.status == 'Authorized' %}✅{% else %}⛔{% endif %} {{ capture.status_ar }}
                            </span>
                        </div>
                        <div class="event-body">
                            <div class="event-heading">
                                <h3>{{ capture.display_name or capture.person_name }}</h3>
                                <time>{{ capture.time_only }}</time>
                            </div>
                            <div class="facts">
                                <div class="fact">
                                    <span>الرقم الجامعي</span>
                                    <strong class="mono">{{ capture.student_id or '—' }}</strong>
                                </div>
                                <div class="fact">
                                    <span>رقم التصريح</span>
                                    <strong class="mono">{{ capture.permit_number or '—' }}</strong>
                                </div>
                                <div class="fact">
                                    <span>التصنيف</span>
                                    <strong>{{ capture.category }}</strong>
                                </div>
                                <div class="fact">
                                    <span>الطابق</span>
                                    <strong>{{ capture.floor or '—' }}</strong>
                                </div>
                            </div>
                            {% if capture.status != 'Authorized' and capture.reason %}
                                <p class="reason">{{ capture.reason }}</p>
                            {% endif %}
                        </div>
                    </article>
                    {% endfor %}
                </div>
            </section>
            {% else %}
            <div class="empty">
                <strong>لا توجد محاولات دخول بعد</strong>
                النظام في وضع الاستعداد بانتظار إشارة المستشعر فوق الصوتي.
            </div>
            {% endfor %}
        </section>
    </main>
</body>
</html>
<script>setTimeout(function(){ window.location.reload(); }, 5000);</script>
"""


@app.route("/")
def home():
    try:
        captures = fetch_recent_events(limit=1000)
    except sqlite3.Error as exc:
        logger.error("Dashboard could not read access logs: %s", exc)
        captures = []

    groups = group_events_by_day(captures)
    requested_day = (request.args.get("day") or "").strip()
    show_all = requested_day.casefold() == "all"
    day_keys = {group["key"] for group in groups}

    if not show_all and requested_day not in day_keys:
        requested_day = groups[0]["key"] if groups else ""

    selected_group = next(
        (group for group in groups if group["key"] == requested_day),
        None,
    )
    if show_all:
        visible_groups = groups
    elif selected_group is not None:
        visible_groups = [selected_group]
    else:
        visible_groups = []

    return render_template_string(
        HTML_TEMPLATE,
        groups=groups,
        visible_groups=visible_groups,
        selected_day=requested_day,
        selected_group=selected_group,
        show_all=show_all,
        week_summary=build_week_summary(groups),
        state=get_state(),
    )


@app.route("/capture_logs/<path:filename>")
@app.route("/captured_logs/<path:filename>")
def get_capture_image(filename: str):
    return send_from_directory(str(CAPTURE_DIR), filename)


@app.route("/start_camera", methods=["GET", "POST"])
@app.route("/trigger", methods=["GET", "POST"])
def start_camera_route():
    """
    ESP32 ultrasonic trigger.

    The camera is idle in standby. This endpoint queues one scan session; the
    trigger service opens the camera, processes one approach, then returns to
    standby. Multiple triggers received while a session is queued coalesce.
    """
    queued = request_scan()
    return (
        jsonify(
            {
                "ok": True,
                "queued": queued,
                "mode": get_state()["mode"],
                "message": "Scan session queued." if queued else "Scan already queued.",
            }
        ),
        202,
    )


@app.route("/reset_security", methods=["GET", "POST"])
def reset_security_route():
    """Clear the unauthorized streak and release lockdown."""
    security = reset_security_state()
    send_esp32_signal("unlock")
    return jsonify({"ok": True, "security": security}), 200


@app.route("/api/status")
def api_status():
    return jsonify(get_state())


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def run_flask_server() -> None:
    logger.info("Dashboard listening on http://%s:%s", DASHBOARD_HOST, DASHBOARD_PORT)
    try:
        app.run(
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except Exception:
        logger.exception("Dashboard server stopped unexpectedly.")


def start_dashboard_thread() -> threading.Thread:
    thread = threading.Thread(target=run_flask_server, name="parkguard-dashboard", daemon=True)
    thread.start()
    return thread


# =============================================================================
# Entry point
# =============================================================================


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParkGuard trigger-driven local gate camera")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="OpenCV camera index (default: 0)")
    parser.add_argument("--no-preview", action="store_true", help="Run without the OpenCV preview window")
    parser.add_argument(
        "--esp32-ip",
        default=ESP32_IP,
        help="ESP32 LAN IP, or empty to disable signalling (env: PARKGUARD_ESP32_IP)",
    )
    parser.add_argument(
        "--rebuild-encodings",
        action="store_true",
        help="Rebuild SQLite face encodings from faces/ or encodings.pickle, then exit",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    global CAMERA_INDEX, ESP32_IP, ENABLE_PREVIEW

    args = parse_args(argv)
    CAMERA_INDEX = args.camera
    ESP32_IP = (args.esp32_ip or "").strip()
    if args.no_preview:
        ENABLE_PREVIEW = False

    configure_logging(args.verbose)

    missing_models = missing_model_files()
    if missing_models:
        logger.error(
            "Missing dlib model file(s): %s. Run: python scripts/download_models.py",
            ", ".join(missing_models),
        )
        return 1

    init_database()

    if args.rebuild_encodings:
        count = rebuild_face_encodings()
        print(f"Rebuilt {count} face encoding(s) in {DB_PATH}")
        return 0

    print("==================================================")
    print(" 🚗 ParkGuard - Triggered Local Gate System")
    print(f" Camera index : {CAMERA_INDEX}")
    print(f" SQLite DB    : {DB_PATH}")
    print(f" Capture logs : {CAPTURE_DIR}")
    print(f" ESP32        : {ESP32_IP or 'disabled (set PARKGUARD_ESP32_IP)'}")
    print(f" TTS greeting : {'on' if TTS_ENABLED else 'off'}")
    print(" Camera state : STANDBY (waiting for ultrasonic trigger)")
    print(" Trigger via  : GET/POST /start_camera or /trigger")
    print(" Press Ctrl+C to stop.")
    print("==================================================")

    start_dashboard_thread()

    recognizer = FaceRecognizer()
    tracker = FaceTracker()
    stop_event = threading.Event()

    try:
        recognizer.initialize()
    except Exception:
        logger.exception("Could not initialize face/permit data; continuing with unknown faces.")

    security = load_security_state()
    update_state(
        known_faces=recognizer.known_count,
        permits=recognizer.permit_count,
        camera_index=CAMERA_INDEX,
        unauthorized_streak=security["unauthorized_streak"],
        lockdown=security["lockdown"],
        mode="standby",
        camera_connected=False,
    )

    try:
        run_trigger_service(recognizer, tracker, stop_event)
    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C).")
    finally:
        stop_event.set()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    logger.info("ParkGuard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
