"""Local SQLite case handling and audit history for the analyst dashboard."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.dataset_paths import BASE_DIR, DATASET

CASE_DB = os.path.join(BASE_DIR, "data", "state", f"{DATASET}_cases.sqlite3")
CASE_STATUSES = ("New", "Investigating", "Escalated", "Resolved", "Benign")
CHECKLIST_ITEMS = (
    "Validate the activity with the account owner or identity team",
    "Review related authentication events and available security alerts",
    "Verify the destination asset and its business criticality",
    "Document the containment decision, approver, and rollback owner",
)


def _connect():
    os.makedirs(os.path.dirname(CASE_DB), exist_ok=True)
    connection = sqlite3.connect(CASE_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            dataset TEXT NOT NULL,
            event_id TEXT NOT NULL,
            status TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            checklist_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (dataset, event_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS case_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            event_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            details_json TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            FOREIGN KEY (dataset, event_id)
                REFERENCES cases (dataset, event_id)
        )
        """
    )
    connection.commit()
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def get_cases(dataset):
    with _connection() as connection:
        rows = connection.execute(
            "SELECT event_id, status, owner, notes, checklist_json, created_at, updated_at "
            "FROM cases WHERE dataset = ?",
            (dataset,),
        ).fetchall()
    cases = {}
    for row in rows:
        case = dict(row)
        case["checklist"] = json.loads(case.pop("checklist_json"))
        cases[case["event_id"]] = case
    return cases


def get_case_history(dataset, event_id):
    with _connection() as connection:
        rows = connection.execute(
            "SELECT action, actor, details_json, occurred_at FROM case_audit "
            "WHERE dataset = ? AND event_id = ? ORDER BY id DESC LIMIT 50",
            (dataset, event_id),
        ).fetchall()
    history = []
    for row in rows:
        entry = dict(row)
        entry["details"] = json.loads(entry.pop("details_json"))
        history.append(entry)
    return history


def save_case(dataset, event_id, status, owner, notes, checklist):
    if status not in CASE_STATUSES:
        raise ValueError(f"Unsupported case status: {status!r}")
    if len(notes) > 10_000:
        raise ValueError("Case notes must be 10,000 characters or fewer.")
    if len(owner) > 120:
        raise ValueError("Analyst name must be 120 characters or fewer.")
    checklist_state = {item: bool(checklist.get(item, False)) for item in CHECKLIST_ITEMS}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _connection() as connection:
        previous = connection.execute(
            "SELECT status, owner, notes, checklist_json, created_at FROM cases "
            "WHERE dataset = ? AND event_id = ?",
            (dataset, event_id),
        ).fetchone()
        created_at = previous["created_at"] if previous else now
        previous_state = (
            {
                "status": previous["status"],
                "owner": previous["owner"],
                "notes": previous["notes"],
                "checklist": json.loads(previous["checklist_json"]),
            }
            if previous else None
        )
        new_state = {
            "status": status,
            "owner": owner.strip(),
            "notes": notes.strip(),
            "checklist": checklist_state,
        }
        connection.execute(
            """
            INSERT INTO cases (
                dataset, event_id, status, owner, notes, checklist_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset, event_id) DO UPDATE SET
                status = excluded.status,
                owner = excluded.owner,
                notes = excluded.notes,
                checklist_json = excluded.checklist_json,
                updated_at = excluded.updated_at
            """,
            (
                dataset,
                event_id,
                status,
                owner.strip(),
                notes.strip(),
                json.dumps(checklist_state, sort_keys=True),
                created_at,
                now,
            ),
        )
        if previous_state != new_state:
            action = "case_created" if previous_state is None else "case_updated"
            connection.execute(
                "INSERT INTO case_audit "
                "(dataset, event_id, action, actor, details_json, occurred_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    dataset,
                    event_id,
                    action,
                    owner.strip() or "unassigned analyst",
                    json.dumps({"before": previous_state, "after": new_state}, sort_keys=True),
                    now,
                ),
            )


def export_cases(dataset):
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT c.event_id, c.status, c.owner, c.notes, c.checklist_json,
                   c.created_at, c.updated_at, COUNT(a.id) AS audit_entries
            FROM cases AS c
            LEFT JOIN case_audit AS a
              ON a.dataset = c.dataset AND a.event_id = c.event_id
            WHERE c.dataset = ?
            GROUP BY c.dataset, c.event_id
            ORDER BY c.updated_at DESC
            """,
            (dataset,),
        ).fetchall()
    exported = []
    for row in rows:
        entry = dict(row)
        entry["checklist"] = json.loads(entry.pop("checklist_json"))
        exported.append(entry)
    return exported
