"""
Optional serial link to the ESP32 gate controller.

The project normally talks to the ESP32 over local HTTP (see main.send_esp32_signal).
Some boards also expose a USB serial command interface. That path is optional:
if pyserial is missing or no port is configured, every call becomes a no-op so
the camera and recognition pipeline keeps running unchanged.

Environment:
    PARKGUARD_SERIAL_PORT   e.g. COM5 (empty disables serial)
    PARKGUARD_SERIAL_BAUD   default 115200
    PARKGUARD_SERIAL_TIMEOUT default 0.5 seconds

Serial commands used by ParkGuard:
    WELCOME Alice
    SERVO OPEN 90
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("parkguard.esp32")

try:  # pyserial is optional; the HTTP path remains the primary transport.
    import serial  # type: ignore
except Exception:  # pragma: no cover - depends on the local environment
    serial = None  # type: ignore[assignment]


_lock = threading.Lock()
_connection: Optional[Any] = None
_failed_port: Optional[str] = None


def _config() -> tuple[str, int, float]:
    port = os.environ.get("PARKGUARD_SERIAL_PORT", "").strip()
    try:
        baud = int(os.environ.get("PARKGUARD_SERIAL_BAUD", "115200"))
    except (TypeError, ValueError):
        baud = 115200
    try:
        timeout = float(os.environ.get("PARKGUARD_SERIAL_TIMEOUT", "0.5"))
    except (TypeError, ValueError):
        timeout = 0.5
    return port, baud, max(0.05, timeout)


def serial_available() -> bool:
    """True when pyserial is installed and a port is configured."""
    port, _, _ = _config()
    return serial is not None and bool(port)


def _open(port: str, baud: int, timeout: float) -> Optional[Any]:
    global _connection, _failed_port
    if _connection is not None:
        return _connection
    if _failed_port == port:
        return None
    try:
        _connection = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        _failed_port = None
        logger.info("ESP32 serial connected on %s @ %s", port, baud)
        return _connection
    except Exception as exc:  # serial.SerialException, OSError, ...
        _failed_port = port
        logger.warning("ESP32 serial unavailable on %s: %s", port, exc)
        return None


def send_command(command: str) -> bool:
    """Write one newline-terminated command to the ESP32 serial port."""
    text = str(command).strip()
    if not text or not serial_available():
        return False

    port, baud, timeout = _config()
    with _lock:
        connection = _open(port, baud, timeout)
        if connection is None:
            return False
        try:
            connection.write(f"{text}\n".encode("utf-8"))
            connection.flush()
            logger.info("ESP32 serial <- %s", text)
            return True
        except Exception as exc:
            global _connection, _failed_port
            logger.warning("ESP32 serial write failed on %s: %s", port, exc)
            try:
                connection.close()
            except Exception:
                pass
            _connection = None
            _failed_port = port
            return False


def close() -> None:
    """Close the serial link, if one is open."""
    global _connection, _failed_port
    with _lock:
        if _connection is not None:
            try:
                _connection.close()
            except Exception:
                pass
        _connection = None
        _failed_port = None
