#!/usr/bin/env python3
"""
Download the dlib model files used by ParkGuard.

The model files are intentionally not committed to Git: they are large,
third-party binary artifacts. Run this script once after cloning:

    python scripts/download_models.py

The files are written next to ``main.py`` so ``face_models.py`` can find them.
"""

from __future__ import annotations

import argparse
import bz2
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DLIB_FILES_URL = "https://dlib.net/files"

# destination filename -> compressed filename on dlib.net
MODEL_FILES: dict[str, str] = {
    "shape_predictor_68_face_landmarks.dat": "shape_predictor_68_face_landmarks.dat.bz2",
    "shape_predictor_5_face_landmarks.dat": "shape_predictor_5_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": "dlib_face_recognition_resnet_model_v1.dat.bz2",
    "mmod_human_face_detector.dat": "mmod_human_face_detector.dat.bz2",
}


def _format_size(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def download_file(url: str, destination: Path) -> None:
    """Download one compressed model and extract it atomically."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ParkGuard-model-downloader/1.0"},
    )
    print(f"[INFO] Downloading {url}")

    with tempfile.NamedTemporaryFile(
        prefix=f"{destination.name}.",
        suffix=".bz2",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                last_report = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded - last_report >= 10 * 1024 * 1024:
                        last_report = downloaded
                        print(
                            f"       {_format_size(downloaded)} / {_format_size(total)} "
                            f"({downloaded * 100 // total}%)"
                        )

            print(f"[INFO] Extracting {destination.name} ...")
            with bz2.open(temporary_path, "rb") as compressed:
                with open(destination, "wb") as output:
                    shutil.copyfileobj(compressed, output)

            if destination.stat().st_size == 0:
                destination.unlink(missing_ok=True)
                raise RuntimeError("downloaded model is empty")

            print(f"[OK]   {destination.name} ({_format_size(destination.stat().st_size)})")
        finally:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ParkGuard dlib models")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace model files that already exist",
    )
    args = parser.parse_args()

    missing = []
    for filename, encoded_name in MODEL_FILES.items():
        destination = PROJECT_DIR / filename
        if destination.is_file() and not args.force:
            print(f"[SKIP] {filename} already exists")
            continue
        missing.append((filename, encoded_name, destination))

    for filename, encoded_name, destination in missing:
        try:
            download_file(f"{DLIB_FILES_URL}/{encoded_name}", destination)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            print(f"[ERROR] Could not download {filename}: {exc}", file=sys.stderr)
            return 1

    print("\nAll model files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
