"""
Train ParkGuard face encodings from local dataset subfolders.

Expected layout (either directory name works):

    faces/
        alice/
            photo1.jpg
            photo2.jpg
        bob/
            photo1.jpg
            photo2.jpg

or the legacy:

    dataset_faces/
        alice/
        bob/

The folder name is the enrolled identity. It must match car_owner in the
permits table exactly.

This script uses the local dlib models shipped in the project (no cloud calls)
and writes:
    encodings.pickle   name + 128-d encoding pairs
    park_guard.db      face_encodings rows used by main.py

Run:
    python train_faces.py
    python train_faces.py --dataset dataset_faces
    python train_faces.py --no-db
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _import_face_recognition():
    from face_models import import_face_recognition

    return import_face_recognition()


def resolve_dataset_dir(explicit: Optional[str] = None) -> Path:
    """Pick the requested dataset path, or the first existing known location."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.is_dir():
            raise SystemExit(f"[ERROR] Dataset directory not found: {path}")
        return path

    for candidate in (PROJECT_DIR / "faces", PROJECT_DIR / "dataset_faces"):
        if candidate.is_dir():
            return candidate

    raise SystemExit(
        "[ERROR] No dataset directory found. Create 'faces' (or 'dataset_faces') "
        "with one subfolder per person."
    )


def _largest_face(locations: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
    if not locations:
        return None

    def area(location: tuple[int, int, int, int]) -> int:
        top, right, bottom, left = location
        return max(0, bottom - top) * max(0, right - left)

    return max(locations, key=area)


def encode_dataset(
    dataset_dir: Path,
) -> tuple[list[str], list[np.ndarray], list[str], dict[str, int]]:
    """Return names, encodings, source labels, and a per-person success count."""
    face_recognition = _import_face_recognition()

    person_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not person_dirs:
        raise SystemExit(f"[ERROR] No person subfolders found in {dataset_dir}")

    names: list[str] = []
    encodings: list[np.ndarray] = []
    sources: list[str] = []
    counts: dict[str, int] = {}

    print(f"[INFO] Reading face images from: {dataset_dir}")
    for person_dir in person_dirs:
        image_paths = sorted(
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[person_dir.name] = 0

        if not image_paths:
            print(f"[WARNING] No images in {person_dir}")
            continue

        for image_path in image_paths:
            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(image)
                largest = _largest_face(locations)
                if largest is None:
                    print(f"[WARNING] No face detected: {person_dir.name}/{image_path.name}")
                    continue

                found = face_recognition.face_encodings(image, [largest])
                if not found:
                    print(f"[WARNING] Could not encode: {person_dir.name}/{image_path.name}")
                    continue

                names.append(person_dir.name)
                encodings.append(np.asarray(found[0], dtype=np.float64))
                sources.append(f"{image_path.name}#face0")
                counts[person_dir.name] += 1
                print(f"[SUCCESS] Encoded {person_dir.name}/{image_path.name}")
            except Exception as exc:  # dlib can raise various runtime errors
                print(f"[WARNING] Skipping {image_path.name}: {exc}")

    return names, encodings, sources, counts


def save_pickle(path: Path, names: list[str], encodings: list[np.ndarray]) -> None:
    payload = {"names": names, "encodings": encodings}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SUCCESS] Saved {len(encodings)} encoding(s) to {path}")


def save_sqlite(
    db_path: Path,
    names: list[str],
    encodings: list[np.ndarray],
    sources: list[str],
) -> None:
    """Replace each trained owner's rows in the face_encodings table."""
    created_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    owners = sorted(set(names))

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_encodings (
                encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_owner TEXT NOT NULL,
                encoding BLOB NOT NULL,
                source_image TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(car_owner, source_image)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_face_encodings_owner
            ON face_encodings(car_owner)
            """
        )

        for owner in owners:
            conn.execute("DELETE FROM face_encodings WHERE car_owner = ?", (owner,))

        for index, (name, encoding) in enumerate(zip(names, encodings)):
            conn.execute(
                """
                INSERT OR REPLACE INTO face_encodings
                    (car_owner, encoding, source_image, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name,
                    sqlite3.Binary(np.asarray(encoding, dtype=np.float64).tobytes()),
                    sources[index],
                    created_at,
                ),
            )

        conn.commit()

    print(f"[SUCCESS] Updated {len(encodings)} encoding(s) in {db_path}")


def train(
    dataset_dir: Path,
    output_path: Path,
    db_path: Path,
    write_db: bool = True,
) -> int:
    names, encodings, sources, counts = encode_dataset(dataset_dir)

    if not encodings:
        print("[ERROR] No faces were encoded; nothing was saved.")
        return 1

    save_pickle(output_path, names, encodings)

    if write_db:
        save_sqlite(db_path, names, encodings, sources)

    print("[INFO] Per-person encodings:")
    for person, count in counts.items():
        status = "OK" if count else "NO FACES"
        print(f"       {person}: {count} ({status})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train local ParkGuard face encodings from dataset subfolders"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory (default: faces, falling back to dataset_faces)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "encodings.pickle"),
        help="Output pickle path (default: encodings.pickle)",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_DIR / "park_guard.db"),
        help="SQLite database path (default: park_guard.db)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Only write encodings.pickle; do not update SQLite",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    db_path = Path(args.db).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path

    return train(
        dataset_dir=dataset_dir,
        output_path=output_path,
        db_path=db_path,
        write_db=not args.no_db,
    )


if __name__ == "__main__":
    raise SystemExit(main())
