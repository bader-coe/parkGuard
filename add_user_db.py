"""
Add, update, or seed parking permits in park_guard.db.

Permits must be exactly four digits:
    1xxx -> resident / upper floor
    0xxx -> visitor / ground floor

The owner name must match the folder name under faces (or dataset_faces)
exactly, because recognition joins on that name. Display names are the
human-readable names shown on the dashboard and sent to the ESP32.

Examples:
    python add_user_db.py --seed
    python add_user_db.py --owner alice --permit 1001 \
        --student-id 30000001 --display-name "Alice Example"
    python add_user_db.py --owner visitor-1 --permit 0456 --student-id 30000002
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from contextlib import closing
from typing import Optional

# Demo rows created by ``--seed``. Replace these with your own enrolled users;
# owner names must match the faces / dataset_faces subfolder names.
SEED_USERS = (
    {
        "owner": "demo-resident",
        "display_name": "Demo Resident",
        "permit": "1000",
        "student_id": "30000001",
    },
    {
        "owner": "demo-visitor",
        "display_name": "Demo Visitor",
        "permit": "0001",
        "student_id": "30000002",
    },
)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _humanize(value: str) -> str:
    if not value or value.casefold() == "unknown":
        return "Unknown"
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", value) if part)


def display_name_for(owner: str, explicit: Optional[str] = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return _humanize(owner)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS permits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_owner TEXT NOT NULL,
            permit_number TEXT NOT NULL,
            student_id TEXT,
            display_name TEXT
        )
        """
    )
    _ensure_column(conn, "permits", "student_id", "TEXT")
    _ensure_column(conn, "permits", "display_name", "TEXT")

    # access_logs may not exist yet on a fresh install; main.py creates it.
    has_logs = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='access_logs'"
    ).fetchone()
    if has_logs:
        _ensure_column(conn, "access_logs", "student_id", "TEXT")
        _ensure_column(conn, "access_logs", "display_name", "TEXT")


def backfill_access_log_metadata(conn: sqlite3.Connection) -> int:
    """Fill student_id/display_name on existing logs from the permit records."""
    has_logs = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='access_logs'"
    ).fetchone()
    if not has_logs:
        return 0

    cursor = conn.execute(
        """
        UPDATE access_logs
        SET student_id = COALESCE(
                student_id,
                (SELECT p.student_id FROM permits p
                 WHERE lower(p.car_owner) = lower(access_logs.person_name)
                   AND p.student_id IS NOT NULL
                 ORDER BY p.id DESC LIMIT 1)
            ),
            display_name = COALESCE(
                display_name,
                (SELECT p.display_name FROM permits p
                 WHERE lower(p.car_owner) = lower(access_logs.person_name)
                   AND p.display_name IS NOT NULL
                 ORDER BY p.id DESC LIMIT 1)
            )
        WHERE student_id IS NULL OR display_name IS NULL
        """
    )
    return cursor.rowcount


def _database_path() -> str:
    """Use PARKGUARD_DB_PATH when set, otherwise the project database."""
    raw = os.environ.get("PARKGUARD_DB_PATH", "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "park_guard.db")


def add_permit(
    owner_name: str,
    permit_no: str,
    student_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    owner = owner_name.strip()
    permit = permit_no.strip()
    student = (student_id or "").strip() or None
    readable = display_name_for(owner, display_name)

    if not owner:
        raise SystemExit("[ERROR] Owner name must not be empty.")

    if re.fullmatch(r"[01]\d{3}", permit) is None:
        raise SystemExit(
            "[ERROR] Permit must be exactly 4 digits starting with 1 "
            "(resident) or 0 (visitor), for example 1000 or 0001."
        )

    if student is not None and re.fullmatch(r"\d{8}", student) is None:
        raise SystemExit("[ERROR] Student ID must be exactly 8 digits, for example 30000000.")

    db_path = _database_path()

    with closing(sqlite3.connect(db_path)) as conn:
        ensure_schema(conn)

        cursor = conn.execute("SELECT id FROM permits WHERE car_owner = ?", (owner,))
        row = cursor.fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO permits
                    (car_owner, permit_number, student_id, display_name)
                VALUES (?, ?, ?, ?)
                """,
                (owner, permit, student, readable),
            )
            message = (
                f"[SUCCESS] Added '{owner}' | name={readable} | "
                f"student={student or '-'} | permit={permit}"
            )
        else:
            conn.execute(
                """
                UPDATE permits
                SET permit_number = ?, student_id = ?, display_name = ?
                WHERE car_owner = ?
                """,
                (permit, student, readable, owner),
            )
            message = (
                f"[SUCCESS] Updated '{owner}' | name={readable} | "
                f"student={student or '-'} | permit={permit}"
            )

        backfill_access_log_metadata(conn)
        conn.commit()

    kind = "resident / upper floor" if permit.startswith("1") else "visitor / ground floor"
    print(message)
    print(f"[INFO] {permit} = {kind}")


def seed_users() -> None:
    print(f"[INFO] Seeding {len(SEED_USERS)} permit record(s)...")
    for user in SEED_USERS:
        add_permit(
            user["owner"],
            user["permit"],
            user["student_id"],
            user["display_name"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add, update, or seed ParkGuard permits")
    parser.add_argument("--seed", action="store_true", help="Seed the demo users")
    parser.add_argument(
        "--owner",
        default=None,
        help="faces folder name, for example alice",
    )
    parser.add_argument(
        "--permit",
        default=None,
        help="4-digit permit: 1xxx resident, 0xxx visitor",
    )
    parser.add_argument(
        "--student-id",
        default=None,
        help="8-digit student ID, for example 30000001",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help='Human-readable name, for example "Alice Example"',
    )
    args = parser.parse_args()

    if args.seed:
        seed_users()
    elif not args.owner or not args.permit:
        parser.error("--owner and --permit are required unless --seed is used")
    else:
        add_permit(args.owner, args.permit, args.student_id, args.display_name)


if __name__ == "__main__":
    main()
