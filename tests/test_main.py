"""Behavior tests for the trigger-driven local ParkGuard pipeline.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from urllib.parse import quote
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ROOT = tempfile.mkdtemp(prefix="parkguard_tests_")

# These must be set before importing main because the module reads its paths at import time.
os.environ["PARKGUARD_DB_PATH"] = os.path.join(TEST_ROOT, "park_guard_test.db")
os.environ["PARKGUARD_CAPTURE_DIR"] = os.path.join(TEST_ROOT, "capture_logs")
os.environ["PARKGUARD_DATASET_DIR"] = os.path.join(TEST_ROOT, "dataset")
os.environ["PARKGUARD_ENCODINGS_PICKLE"] = os.path.join(TEST_ROOT, "missing_encodings.pickle")
os.environ["PARKGUARD_PREVIEW"] = "0"
os.environ["PARKGUARD_ESP32_IP"] = ""
os.environ["PARKGUARD_DB_REFRESH_SECONDS"] = "0"
os.environ["PARKGUARD_TTS"] = "0"

import main  # noqa: E402


def _authorized(name: str = "alice", permit: str = "1001") -> "main.AuthorizationResult":
    return main.AuthorizationResult(
        name=name,
        authorized=True,
        distance=0.1,
        permit=main.validate_permit(permit),
        reason="Authorized",
    )


def _unknown() -> "main.AuthorizationResult":
    return main.AuthorizationResult(
        name="Unknown",
        authorized=False,
        distance=0.9,
        permit=None,
        reason="Unknown face",
    )


def _track(result, bbox=(100, 120, 220, 260), track_id=1, now=0.0) -> "main.FaceTrack":
    return main.FaceTrack(
        track_id=track_id,
        bbox=bbox,
        result=result,
        identity_key=(result.name, result.authorized, result.reason),
        identity_since=now,
        last_seen=now,
        created_at=now,
    )


class ParkGuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        main.init_database()
        with closing(sqlite3.connect(str(main.DB_PATH))) as conn:
            conn.execute("DELETE FROM access_logs")
            conn.execute("DELETE FROM face_encodings")
            conn.execute("DELETE FROM permits")
            conn.commit()
        # Keep disk/DB reconciliation tests isolated from earlier captures.
        if main.CAPTURE_DIR.is_dir():
            for path in main.CAPTURE_DIR.iterdir():
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        pass
        main.clear_scan_requests()
        main.reset_security_state()
        main.update_state(last_event=None, mode="standby", camera_connected=False)

    def _add_permit(self, owner: str, permit_number: str) -> None:
        with closing(sqlite3.connect(str(main.DB_PATH))) as conn:
            conn.execute(
                "INSERT INTO permits (car_owner, permit_number) VALUES (?, ?)",
                (owner, permit_number),
            )
            conn.commit()

    def test_permit_validation_4_digit_rules(self) -> None:
        resident = main.validate_permit("1001")
        self.assertTrue(resident.valid)
        self.assertEqual(resident.normalized, "1001")
        self.assertTrue(resident.is_resident)
        self.assertEqual(resident.floor, "Upper")

        visitor = main.validate_permit("0456")
        self.assertTrue(visitor.valid)
        self.assertEqual(visitor.normalized, "0456")
        self.assertFalse(visitor.is_resident)
        self.assertEqual(visitor.floor, "Ground")

        self.assertTrue(main.validate_permit(" 1001 ").valid)

        with_student = main.validate_permit("1000", "30000002", "Bob Example")
        self.assertTrue(with_student.valid)
        self.assertEqual(with_student.student_id, "30000002")
        self.assertEqual(with_student.display_name, "Bob Example")
        self.assertTrue(with_student.is_resident)

        for bad in ("2456", "123", "10011", "KFUPM-2026-001", "", "abcd", None):
            with self.subTest(permit=bad):
                info = main.validate_permit(bad)
                self.assertFalse(info.valid)
                self.assertIsNone(info.normalized)
                self.assertIsNone(info.is_resident)
                self.assertIsNone(info.floor)

    def test_load_permit_map_prefers_valid_duplicate(self) -> None:
        self._add_permit("alice", "KFUPM-2026-001")
        self._add_permit("alice", "1001")
        self._add_permit("visitor", "0456")

        permits = main.load_permit_map()
        self.assertTrue(permits["alice"].valid)
        self.assertEqual(permits["alice"].normalized, "1001")
        self.assertFalse(permits["visitor"].is_resident)
        self.assertEqual(permits["visitor"].floor, "Ground")

    def test_face_encoding_roundtrip_and_authorized_match(self) -> None:
        self._add_permit("alice", "1001")
        encoding = np.linspace(0.0, 1.0, 128, dtype=np.float64)
        self.assertTrue(main.store_face_encoding("alice", encoding, "unit-test#face0"))

        encodings, names = main.load_face_encodings()
        self.assertEqual(encodings.shape, (1, 128))
        self.assertEqual(names, ["alice"])

        recognizer = main.FaceRecognizer()
        recognizer.refresh(force=True)
        result = recognizer.identify(encoding)
        self.assertTrue(result.authorized)
        self.assertEqual(result.name, "alice")
        self.assertEqual(result.permit.normalized, "1001")

    def test_known_face_with_invalid_permit_is_unknown(self) -> None:
        self._add_permit("alice", "KFUPM-2026-001")
        encoding = np.zeros(128, dtype=np.float64)
        main.store_face_encoding("alice", encoding, "unit-test#face0")

        recognizer = main.FaceRecognizer()
        recognizer.refresh(force=True)
        result = recognizer.identify(encoding)

        self.assertFalse(result.authorized)
        self.assertEqual(result.overlay_label, "Unknown")
        self.assertIn("4", result.reason)

    def test_known_face_without_permit_is_unknown(self) -> None:
        encoding = np.full(128, 0.25, dtype=np.float64)
        main.store_face_encoding("stranger", encoding, "unit-test#face0")

        recognizer = main.FaceRecognizer()
        recognizer.refresh(force=True)
        result = recognizer.identify(encoding)

        self.assertFalse(result.authorized)
        self.assertEqual(result.overlay_label, "Unknown")
        self.assertIn("permit", result.reason.lower())

    def test_unknown_face_is_red_with_label_above(self) -> None:
        track = _track(_unknown(), bbox=(100, 120, 220, 260))
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        main.annotate_frame(frame, [track])

        # Red border (BGR) at the top-left corner of the box.
        self.assertTrue(np.all(frame[120, 100] == np.array([0, 0, 255])))
        # The filled label background above the box must also be red.
        label_region = frame[70:120, 100:260]
        self.assertTrue(np.any(np.all(label_region == np.array([0, 0, 255]), axis=2)))

        with mock.patch.object(main.cv2, "putText", wraps=main.cv2.putText) as put_text:
            main.annotate_frame(np.full((480, 640, 3), 255, dtype=np.uint8), [track])
        rendered = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Unknown", rendered)

    def test_authorized_face_is_green(self) -> None:
        track = _track(_authorized())
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        main.annotate_frame(frame, [track])

        self.assertTrue(np.all(frame[120, 100] == np.array([0, 255, 0])))

    def test_tracker_hold_and_no_repeated_capture(self) -> None:
        tracker = main.FaceTracker()
        detection = main.Detection(bbox=(100, 120, 220, 260), result=_authorized())

        track = tracker.update([detection], now=0.0)[0]
        self.assertFalse(track.ready_for_capture(0.5))

        for moment in (0.5, 1.0, 1.5, 2.0, 2.5):
            track = tracker.update([detection], now=moment)[0]
            self.assertFalse(track.ready_for_capture(moment))

        track = tracker.update([detection], now=3.0)[0]
        self.assertAlmostEqual(track.visible_seconds, 3.0, places=6)
        self.assertTrue(track.ready_for_capture(3.0))

        track.captured = True
        self.assertFalse(track.ready_for_capture(5.0))

    def test_tracker_unknown_delay_and_identity_reset(self) -> None:
        tracker = main.FaceTracker()
        detection = main.Detection(bbox=(100, 120, 220, 260), result=_unknown())

        track = tracker.update([detection], now=0.0)[0]
        for moment in (0.5, 1.0, 1.5, 2.0, 2.5):
            track = tracker.update([detection], now=moment)[0]
            self.assertFalse(track.ready_for_capture(moment))
        track = tracker.update([detection], now=3.0)[0]
        self.assertTrue(track.ready_for_capture(3.0))

        other = main.Detection(
            bbox=(100, 120, 220, 260),
            result=_authorized(name="other", permit="0456"),
        )
        track = tracker.update([other], now=3.5)[0]
        self.assertEqual(track.identity_since, 3.5)
        self.assertEqual(track.visible_seconds, 0.0)
        for moment in (4.0, 4.5, 5.0, 5.5, 6.0):
            track = tracker.update([other], now=moment)[0]
            self.assertFalse(track.ready_for_capture(moment))
        track = tracker.update([other], now=6.5)[0]
        self.assertTrue(track.ready_for_capture(6.5))

    def test_stability_countdown_pauses_when_face_is_absent_or_unclear(self) -> None:
        tracker = main.FaceTracker()
        clear_detection = main.Detection(
            bbox=(100, 120, 220, 260), result=_authorized()
        )
        unclear_detection = main.Detection(
            bbox=(100, 120, 220, 260), result=_authorized(), clear=False
        )

        track = tracker.update([clear_detection], now=0.0)[0]
        for moment in (0.5, 1.0):
            track = tracker.update([clear_detection], now=moment)[0]
        self.assertAlmostEqual(track.visible_seconds, 1.0, places=6)

        # A covered/unclear face pauses the countdown even though a track exists.
        for moment in (1.5, 2.0, 2.5):
            track = tracker.update([unclear_detection], now=moment)[0]
            self.assertFalse(track.ready_for_capture(moment))
        self.assertAlmostEqual(track.visible_seconds, 1.0, places=6)

        # A long absence must not grant the missing seconds either.
        track = tracker.update([clear_detection], now=4.0)[0]
        self.assertAlmostEqual(track.visible_seconds, 1.0, places=6)
        self.assertFalse(track.ready_for_capture(4.0))

        # Clear visibility resumes and eventually completes the 3-second hold.
        for moment in (4.5, 5.0, 5.5):
            track = tracker.update([clear_detection], now=moment)[0]
            self.assertFalse(track.ready_for_capture(moment))
        track = tracker.update([clear_detection], now=6.0)[0]
        self.assertTrue(track.ready_for_capture(6.0))

    def test_save_unknown_capture_uses_precise_timestamp_and_db(self) -> None:
        track = _track(_unknown())
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        filename = main.save_capture(frame, track)

        self.assertIsNotNone(filename)
        self.assertRegex(
            filename,
            r"^capture_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}_Unknown_Unauthorized_t\d+\.jpg$",
        )
        self.assertTrue((main.CAPTURE_DIR / filename).is_file())
        self.assertTrue(track.captured)

        events = main.fetch_recent_events(limit=10)
        self.assertEqual(events[0]["status"], "Unauthorized")
        self.assertEqual(events[0]["person_name"], "Unknown")
        self.assertEqual(events[0]["image_filename"], filename)

    def test_save_authorized_capture_stores_permit_details(self) -> None:
        track = _track(_authorized(name="alice", permit="1001"))
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        filename = main.save_capture(frame, track)

        self.assertIn("_Authorized", filename)
        events = main.fetch_recent_events(limit=10)
        self.assertEqual(events[0]["status"], "Authorized")
        self.assertEqual(events[0]["permit_number"], "1001")
        self.assertEqual(events[0]["is_resident"], 1)
        self.assertEqual(events[0]["floor"], "Upper")

    def test_authorized_capture_stores_student_id_and_dashboard_column(self) -> None:
        permit = main.validate_permit("0001", "30000001", "Alice Example")
        result = main.AuthorizationResult(
            name="alice",
            authorized=True,
            distance=0.05,
            permit=permit,
            reason="Authorized",
        )
        track = _track(result)

        filename = main.save_capture(
            np.full((480, 640, 3), 255, dtype=np.uint8), track
        )

        self.assertIn("_Authorized", filename)
        event = main.fetch_recent_events(limit=1)[0]
        self.assertEqual(event["person_name"], "alice")
        self.assertEqual(event["display_name"], "Alice Example")
        self.assertEqual(event["student_id"], "30000001")
        self.assertEqual(event["permit_number"], "0001")
        self.assertEqual(event["is_resident"], 0)
        self.assertEqual(event["floor"], "Ground")

        home = main.app.test_client().get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("الرقم الجامعي".encode("utf-8"), home.data)
        self.assertIn(b"30000001", home.data)
        self.assertIn(b"Alice Example", home.data)
        self.assertIn(b"event-card", home.data)
        self.assertIn(b"day-header", home.data)

    def test_authorized_capture_greets_user_and_opens_servo(self) -> None:
        permit = main.validate_permit("1000", "30000002", "Bob Example")
        result = main.AuthorizationResult(
            name="bob",
            authorized=True,
            distance=0.05,
            permit=permit,
            reason="Authorized",
        )

        with mock.patch.object(main, "send_esp32_signal") as signal, mock.patch.object(
            main.esp32_comm, "send_command"
        ) as serial:
            main.save_capture(
                np.full((480, 640, 3), 255, dtype=np.uint8), _track(result)
            )

        endpoints = [call.args[0] for call in signal.call_args_list]
        self.assertIn("greeting", endpoints)
        self.assertIn("servo", endpoints)

        greeting_call = next(
            call for call in signal.call_args_list if call.args[0] == "greeting"
        )
        self.assertEqual(greeting_call.kwargs["params"]["message"], "Welcome Bob")
        self.assertEqual(greeting_call.kwargs["params"]["name"], "Bob")

        serial.assert_any_call("WELCOME Bob")
        serial.assert_any_call(f"SERVO OPEN {main.SERVO_OPEN_ANGLE}")

    def test_day_grouping_and_weekday_labels(self) -> None:
        events = [
            {
                "day_key": "2026-10-25",
                "weekday_en": "Sunday",
                "weekday_ar": "الأحد",
                "date_label": "2026-10-25",
                "person_name": "alice",
                "status": "Authorized",
            },
            {
                "day_key": "2026-10-25",
                "weekday_en": "Sunday",
                "weekday_ar": "الأحد",
                "date_label": "2026-10-25",
                "person_name": "bob",
                "status": "Unauthorized",
            },
            {
                "day_key": "2026-10-24",
                "weekday_en": "Saturday",
                "weekday_ar": "السبت",
                "date_label": "2026-10-24",
                "person_name": "alice",
                "status": "Authorized",
            },
        ]

        groups = main.group_events_by_day(events)
        summary = main.build_week_summary(groups)

        self.assertEqual([group["weekday_en"] for group in groups], ["Sunday", "Saturday"])
        self.assertEqual(len(groups[0]["events"]), 2)
        self.assertEqual(groups[0]["authorized"], 1)
        self.assertEqual(groups[0]["unauthorized"], 1)
        self.assertEqual(groups[1]["weekday_ar"], "السبت")
        self.assertEqual(summary, {"days": 2, "total": 3, "authorized": 2, "unauthorized": 1})

    def test_dashboard_weekly_summary_and_day_switch(self) -> None:
        main.record_access_event(
            event_time="2026-10-24T10:00:00+00:00",
            person_name="alice",
            permit_number="0001",
            is_resident=False,
            floor="Ground",
            status="Authorized",
            reason="Authorized.",
            image_path=None,
            student_id="30000001",
            display_name="Alice Example",
        )
        main.record_access_event(
            event_time="2026-10-25T11:00:00+00:00",
            person_name="bob",
            permit_number="1000",
            is_resident=True,
            floor="Upper",
            status="Authorized",
            reason="Authorized.",
            image_path=None,
            student_id="30000002",
            display_name="Bob Example",
        )

        client = main.app.test_client()

        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        home_text = home.get_data(as_text=True)
        self.assertIn("الملخص الأسبوعي", home_text)
        self.assertIn("summary-card", home_text)
        self.assertIn("Bob Example", home_text)
        self.assertNotIn("Alice Example", home_text)

        older_day = client.get("/?day=2026-10-24")
        older_text = older_day.get_data(as_text=True)
        self.assertIn("Alice Example", older_text)
        self.assertNotIn("Bob Example", older_text)

        all_days = client.get("/?day=all")
        all_text = all_days.get_data(as_text=True)
        self.assertIn("Alice Example", all_text)
        self.assertIn("Bob Example", all_text)

    def test_dashboard_title_is_exactly_parkguard(self) -> None:
        response = main.app.test_client().get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>ParkGuard</title>", text)
        self.assertIn("<h1>ParkGuard</h1>", text)
        self.assertNotIn("ParkGuard v2", text)

    def test_fetch_recent_events_drops_ghosts_and_recovers_orphans(self) -> None:
        ghost_name = (
            "capture_2026-10-25_09-00-00-000000_ghost_Authorized_t1.jpg"
        )
        main.record_access_event(
            event_time="2026-10-25T09:00:00+00:00",
            person_name="ghost",
            permit_number="1001",
            is_resident=True,
            floor="Upper",
            status="Authorized",
            reason="Authorized.",
            image_path=ghost_name,
            display_name="Ghost User",
        )

        orphan_name = (
            "capture_2026-10-25_10-00-00-123456_alice_Authorized_t7.jpg"
        )
        (main.CAPTURE_DIR / orphan_name).write_bytes(b"orphan-image")

        real_name = (
            "capture_2026-10-25_11-00-00-000000_bob_Unauthorized_t8.jpg"
        )
        (main.CAPTURE_DIR / real_name).write_bytes(b"real-image")
        main.record_access_event(
            event_time="2026-10-25T11:00:00+00:00",
            person_name="Unknown",
            permit_number=None,
            is_resident=None,
            floor=None,
            status="Unauthorized",
            reason="Unknown face.",
            image_path=real_name,
        )

        events = main.fetch_recent_events(limit=10)
        names = [event["image_filename"] for event in events]

        self.assertNotIn(ghost_name, names)
        self.assertIn(orphan_name, names)
        self.assertIn(real_name, names)
        self.assertEqual(names.count(real_name), 1)

        orphan = next(event for event in events if event["image_filename"] == orphan_name)
        self.assertEqual(orphan["status"], "Authorized")
        self.assertEqual(orphan["person_name"], "alice")
        self.assertEqual(orphan["display_name"], "Alice")
        self.assertEqual(orphan["day_key"], "2026-10-25")

    def test_face_clear_check_rejects_small_or_flat_crops(self) -> None:
        encoding = np.zeros(128, dtype=np.float64)
        flat = np.full((480, 640, 3), 255, dtype=np.uint8)
        noisy = np.random.default_rng(3).integers(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )

        self.assertFalse(main._face_is_clear(flat, (100, 100, 200, 200), encoding))
        self.assertTrue(main._face_is_clear(noisy, (100, 100, 200, 200), encoding))
        self.assertFalse(main._face_is_clear(noisy, (100, 100, 120, 120), encoding))
        self.assertFalse(
            main._face_is_clear(noisy, (100, 100, 200, 200), np.zeros(64, dtype=np.float64))
        )

    def test_speak_welcome_uses_pyttsx3_engine(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.said: list[str] = []
                self.ran = 0

            def setProperty(self, name: str, value: object) -> None:
                pass

            def say(self, text: str) -> None:
                self.said.append(text)

            def runAndWait(self) -> None:
                self.ran += 1

        class FakePyttsx3:
            def __init__(self) -> None:
                self.engine = FakeEngine()

            def init(self) -> FakeEngine:
                return self.engine

        fake = FakePyttsx3()
        with mock.patch.object(main, "pyttsx3", fake), mock.patch.object(
            main, "_tts_engine", None
        ), mock.patch.object(main, "TTS_ENABLED", True):
            self.assertTrue(main.speak_welcome("Alice", wait=True))

        self.assertEqual(fake.engine.said, ["Welcome Alice"])
        self.assertEqual(fake.engine.ran, 1)

    def test_announce_authorized_triggers_tts_and_greeting(self) -> None:
        result = main.AuthorizationResult(
            name="bob",
            authorized=True,
            distance=0.05,
            permit=main.validate_permit("1000", "30000002", "Bob Example"),
            reason="Authorized",
        )

        with mock.patch.object(main, "speak_welcome") as speak, mock.patch.object(
            main, "send_esp32_signal"
        ) as signal, mock.patch.object(main.esp32_comm, "send_command"):
            message = main.announce_authorized(result)

        self.assertEqual(message, "Welcome Bob")
        speak.assert_called_once_with("Bob")
        endpoints = [call.args[0] for call in signal.call_args_list]
        self.assertIn("greeting", endpoints)
        self.assertIn("servo", endpoints)

    def test_serial_is_optional_when_port_is_not_configured(self) -> None:
        with mock.patch.dict(os.environ, {"PARKGUARD_SERIAL_PORT": ""}):
            self.assertFalse(main.esp32_comm.serial_available())
            self.assertFalse(main.esp32_comm.send_command("WELCOME Alice"))

    def test_seed_definitions_match_demo_users(self) -> None:
        import add_user_db

        seeded = {user["owner"]: user for user in add_user_db.SEED_USERS}

        self.assertEqual(seeded["demo-resident"]["student_id"], "30000001")
        self.assertEqual(seeded["demo-resident"]["permit"], "1000")
        self.assertEqual(seeded["demo-resident"]["display_name"], "Demo Resident")
        self.assertEqual(seeded["demo-visitor"]["student_id"], "30000002")
        self.assertEqual(seeded["demo-visitor"]["permit"], "0001")
        self.assertEqual(seeded["demo-visitor"]["display_name"], "Demo Visitor")
        self.assertFalse(main.validate_permit("0001").is_resident)
        self.assertTrue(main.validate_permit("1000").is_resident)

    def test_dashboard_and_status_endpoints(self) -> None:
        track = _track(_unknown())
        filename = main.save_capture(np.full((480, 640, 3), 255, dtype=np.uint8), track)
        client = main.app.test_client()

        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"ParkGuard", home.data)

        status = client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        self.assertIn("camera_connected", status.get_json())

        image = client.get(f"/capture_logs/{quote(filename)}")
        self.assertEqual(image.status_code, 200)
        self.assertGreater(len(image.data), 100)

    def test_process_frame_detects_and_holds_authorized_face(self) -> None:
        self._add_permit("alice", "1001")
        encoding = np.linspace(0.0, 1.0, 128, dtype=np.float64)
        main.store_face_encoding("alice", encoding, "unit-test#face0")
        recognizer = main.FaceRecognizer()
        recognizer.refresh(force=True)
        tracker = main.FaceTracker()
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

        with mock.patch.object(main.face_recognition, "face_locations", return_value=[(50, 150, 120, 70)]), mock.patch.object(
            main.face_recognition, "face_encodings", return_value=[encoding]
        ):
            _, due_at_start = main.process_frame(frame, recognizer, tracker, now=0.0)
            for moment in (0.5, 1.0, 1.5, 2.0, 2.5):
                _, due_before_hold = main.process_frame(
                    frame, recognizer, tracker, now=moment
                )
                self.assertEqual(due_before_hold, [])
            _, due_after_hold = main.process_frame(frame, recognizer, tracker, now=3.0)

        self.assertEqual(due_at_start, [])
        self.assertEqual(len(due_after_hold), 1)
        self.assertEqual(due_after_hold[0].bbox, (175, 125, 375, 300))
        self.assertTrue(due_after_hold[0].result.authorized)
        self.assertGreaterEqual(due_after_hold[0].visible_seconds, 3.0)

    def test_known_face_with_bad_permit_logs_unknown_and_unauthorized(self) -> None:
        result = main.AuthorizationResult(
            name="alice",
            authorized=False,
            distance=0.2,
            permit=main.validate_permit("KFUPM-2026-001"),
            reason="Permit must be exactly 4 digits (1xxx or 0xxx).",
        )
        track = _track(result)

        filename = main.save_capture(np.full((480, 640, 3), 255, dtype=np.uint8), track)

        # Rendering and logging must agree: the overlay label is "Unknown".
        self.assertIn("_Unknown_Unauthorized", filename)
        events = main.fetch_recent_events(limit=1)
        self.assertEqual(events[0]["person_name"], "Unknown")
        self.assertEqual(events[0]["status"], "Unauthorized")
        self.assertIsNone(events[0]["permit_number"])
        self.assertIn("4 digits", events[0]["reason"])

    def test_first_unauthorized_logs_without_buzzer(self) -> None:
        with mock.patch.object(main, "send_esp32_signal") as signal:
            decision = main.process_access_event(_unknown())

        self.assertEqual(decision.unauthorized_streak, 1)
        self.assertFalse(decision.lockdown)
        self.assertFalse(decision.buzzer)
        self.assertEqual(signal.call_args_list, [mock.call("unauthorized")])

        stored = main.load_security_state()
        self.assertEqual(stored["unauthorized_streak"], 1)
        self.assertFalse(stored["lockdown"])

    def test_repeated_unauthorized_triggers_buzzer_and_lockdown(self) -> None:
        main.process_access_event(_unknown(), send_signals=False)

        with mock.patch.object(main, "send_esp32_signal") as signal:
            decision = main.process_access_event(_unknown())

        self.assertEqual(decision.unauthorized_streak, 2)
        self.assertTrue(decision.buzzer)
        self.assertTrue(decision.lockdown)
        sent = [call.args[0] for call in signal.call_args_list]
        self.assertEqual(sent[0], "unauthorized")
        self.assertIn("buzzer", sent)
        self.assertIn("lockdown", sent)

        stored = main.load_security_state()
        self.assertEqual(stored["unauthorized_streak"], 2)
        self.assertTrue(stored["lockdown"])

    def test_authorized_access_clears_streak_and_lockdown(self) -> None:
        main.process_access_event(_unknown(), send_signals=False)
        main.process_access_event(_unknown(), send_signals=False)

        with mock.patch.object(main, "send_esp32_signal") as signal:
            decision = main.process_access_event(_authorized())

        self.assertEqual(decision.unauthorized_streak, 0)
        self.assertFalse(decision.lockdown)
        self.assertFalse(decision.buzzer)
        sent = [call.args[0] for call in signal.call_args_list]
        self.assertIn("unlock", sent)
        self.assertIn("authorized", sent)

        stored = main.load_security_state()
        self.assertEqual(stored["unauthorized_streak"], 0)
        self.assertFalse(stored["lockdown"])

    def test_reset_security_state_clears_lockdown(self) -> None:
        main.process_access_event(_unknown(), send_signals=False)
        main.process_access_event(_unknown(), send_signals=False)
        self.assertTrue(main.load_security_state()["lockdown"])

        main.reset_security_state()

        stored = main.load_security_state()
        self.assertEqual(stored["unauthorized_streak"], 0)
        self.assertFalse(stored["lockdown"])
        self.assertFalse(main.get_state()["lockdown"])

    def test_scan_session_does_nothing_without_trigger(self) -> None:
        stop_event = main.threading.Event()

        with mock.patch.object(main.cv2, "VideoCapture") as video_capture:
            filename = main.run_pending_scan_session(mock.Mock(), mock.Mock(), stop_event, preview=False)

        self.assertIsNone(filename)
        video_capture.assert_not_called()

    def test_trigger_service_is_idle_until_request(self) -> None:
        stop_event = main.threading.Event()
        runner = mock.Mock(return_value=None)

        with mock.patch.object(main, "run_pending_scan_session", runner), mock.patch.object(
            main, "_open_camera"
        ) as open_camera:
            thread = main.threading.Thread(
                target=main.run_trigger_service,
                args=(mock.Mock(), mock.Mock(), stop_event),
                kwargs={"preview": False},
                daemon=True,
            )
            thread.start()
            time.sleep(0.25)
            self.assertEqual(runner.call_count, 0)
            open_camera.assert_not_called()

            main.request_scan()
            for _ in range(100):
                if runner.call_count:
                    break
                time.sleep(0.02)
            stop_event.set()
            thread.join(timeout=3)

        self.assertEqual(runner.call_count, 1)
        self.assertFalse(thread.is_alive())

    def test_one_camera_session_per_trigger(self) -> None:
        track = _track(_unknown())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        class FakeCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def set(self, *args) -> None:
                return None

            def read(self):
                return True, frame

            def release(self) -> None:
                self.released = True

        fake_capture = FakeCapture()
        process_mock = mock.Mock(side_effect=lambda *a, **k: (frame, [track]))
        save_mock = mock.Mock(return_value="capture.jpg")
        stop_event = main.threading.Event()

        with mock.patch.object(main.cv2, "VideoCapture", return_value=fake_capture), mock.patch.object(
            main.cv2, "destroyAllWindows"
        ), mock.patch.object(main, "process_frame", process_mock), mock.patch.object(main, "save_capture", save_mock):
            self.assertIsNone(main.run_pending_scan_session(mock.Mock(), mock.Mock(), stop_event, preview=False))

            main.request_scan()
            filename = main.run_pending_scan_session(mock.Mock(), mock.Mock(), stop_event, preview=False)

            # A second call without a new trigger must not reopen the camera.
            self.assertIsNone(main.run_pending_scan_session(mock.Mock(), mock.Mock(), stop_event, preview=False))

        self.assertEqual(filename, "capture.jpg")
        self.assertEqual(save_mock.call_count, 1)
        self.assertTrue(fake_capture.released)
        self.assertEqual(main.get_state()["mode"], "standby")

    def test_start_camera_route_queues_trigger(self) -> None:
        main.clear_scan_requests()
        client = main.app.test_client()

        response = client.get("/start_camera")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.get_json()["ok"])
        self.assertTrue(main.has_pending_scan())
        main.clear_scan_requests()

        response = client.post("/trigger")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(main.has_pending_scan())
        main.clear_scan_requests()

    def test_reset_security_route(self) -> None:
        main.process_access_event(_unknown(), send_signals=False)
        main.process_access_event(_unknown(), send_signals=False)
        client = main.app.test_client()

        with mock.patch.object(main, "send_esp32_signal") as signal:
            response = client.post("/reset_security")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["security"]["lockdown"])
        signal.assert_called_once_with("unlock")

    def test_enrollment_from_dataset_writes_encodings_to_sqlite(self) -> None:
        encoding = np.linspace(0.0, 1.0, 128, dtype=np.float64)

        with tempfile.TemporaryDirectory(prefix="parkguard_enroll_") as tmp:
            person_dir = Path(tmp) / "alice"
            person_dir.mkdir(parents=True)
            (person_dir / "photo.jpg").write_bytes(b"fake-image")

            with mock.patch.object(main, "DATASET_DIR", Path(tmp)), mock.patch.object(
                main.face_recognition,
                "load_image_file",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ), mock.patch.object(
                main.face_recognition,
                "face_locations",
                return_value=[(0, 10, 10, 0)],
            ), mock.patch.object(
                main.face_recognition,
                "face_encodings",
                return_value=[encoding],
            ):
                count = main.bootstrap_face_encodings(force=True)

        self.assertEqual(count, 1)
        encodings, names = main.load_face_encodings()
        self.assertEqual(encodings.shape, (1, 128))
        self.assertEqual(names, ["alice"])


def tearDownModule() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
