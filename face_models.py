"""
Local dlib model discovery for ParkGuard.

``face_recognition`` normally imports a separate ``face_recognition_models``
package that expects the dlib model files to be installed globally. ParkGuard
is designed to run offline, so the model files live next to this module and are
downloaded with ``python scripts/download_models.py``.

Importing this module registers :class:`LocalModels` under
``sys.modules["face_recognition_models"]`` so the regular ``face_recognition``
import resolves to the project-local files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent

# The keys match the method names used by the face_recognition_models API.
MODEL_FILES: dict[str, str] = {
    "pose_predictor_model_location": "shape_predictor_68_face_landmarks.dat",
    "pose_predictor_five_point_model_location": "shape_predictor_5_face_landmarks.dat",
    "face_recognition_model_location": "dlib_face_recognition_resnet_model_v1.dat",
    "cnn_face_detector_model_location": "mmod_human_face_detector.dat",
}


class LocalModels:
    """Minimal replacement for the face_recognition_models package."""

    @staticmethod
    def get_models_dir() -> str:
        return str(PROJECT_DIR)

    @staticmethod
    def pose_predictor_model_location() -> str:
        return str(PROJECT_DIR / MODEL_FILES["pose_predictor_model_location"])

    @staticmethod
    def pose_predictor_five_point_model_location() -> str:
        return str(PROJECT_DIR / MODEL_FILES["pose_predictor_five_point_model_location"])

    @staticmethod
    def face_recognition_model_location() -> str:
        return str(PROJECT_DIR / MODEL_FILES["face_recognition_model_location"])

    @staticmethod
    def cnn_face_detector_model_location() -> str:
        return str(PROJECT_DIR / MODEL_FILES["cnn_face_detector_model_location"])


def missing_model_files() -> list[str]:
    """Return the names of model files that are not present in the project."""
    return [name for name in MODEL_FILES.values() if not (PROJECT_DIR / name).is_file()]


def register_local_models() -> None:
    """Make ``import face_recognition`` use the files in this directory."""
    sys.modules["face_recognition_models"] = LocalModels


def import_face_recognition() -> Any:
    """Validate the model files and import face_recognition lazily."""
    missing = missing_model_files()
    if missing:
        raise SystemExit(
            "[ERROR] Missing local dlib model file(s): "
            + ", ".join(missing)
            + "\nRun: python scripts/download_models.py"
        )

    register_local_models()
    import face_recognition  # noqa: PLC0415 - imported after model registration

    return face_recognition


class _LazyFaceRecognition:
    """Module-like proxy that loads dlib only when a function is first called.

    Importing ``face_recognition`` immediately opens all four dlib model files.
    Keeping that work behind a proxy lets the CLI print a friendly missing-model
    message instead of crashing at import time, and lets the unit tests run
    before the large model files have been downloaded.
    """

    def __init__(self) -> None:
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = import_face_recognition()
        return self._module

    def __getattr__(self, name: str) -> Any:
        # Do not trigger a model load for Python introspection (for example
        # ``hasattr(module, "__path__")`` or mock's internal checks).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def lazy(*args: Any, **kwargs: Any) -> Any:
            return getattr(self._load(), name)(*args, **kwargs)

        lazy.__name__ = name
        lazy.__doc__ = f"Load face_recognition and call {name}()."
        return lazy


# Public module-like object used by main.py. Tests and callers can still use
# ``mock.patch.object(face_recognition, "face_locations", ...)`` as usual.
face_recognition: Any = _LazyFaceRecognition()


# Register the local model provider as soon as this module is imported; the
# face_recognition package itself stays lazy until a function is called.
register_local_models()
