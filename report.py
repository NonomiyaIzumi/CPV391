"""
Report module: export attendance records to CSV.
"""
import argparse
import csv
import sqlite3

from config import DB_PATH
from database import init_db, get_attendance_for_session, get_session, get_all_sessions


def export_report(
    conn: sqlite3.Connection,
    session_id: str,
    output_path: str | None = None,
) -> str:
    """
    Export attendance for a session to CSV.

    Returns the path of the created CSV file.
    """
    session = get_session(conn, session_id)
    if session is None:
        print(f"[Report] Session {session_id} not found.")
        return ""

    records = get_attendance_for_session(conn, session_id)

    if output_path is None:
        class_name = session.get("class_name", "class")
        start = session.get("start_time", "unknown")
        safe_time = start.replace(":", "-").replace(" ", "_")[:19]
        output_path = f"attendance_{class_name}_{safe_time}.csv"

    fieldnames = [
        "student_id",
        "name",
        "first_seen",
        "last_seen",
        "status",
        "checkout_time",
        "confidence",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    print(f"[Report] Exported {len(records)} records to {output_path}")
    return output_path


def print_summary(conn: sqlite3.Connection, session_id: str):
    """Print a summary of the attendance session to console."""
    session = get_session(conn, session_id)
    if session is None:
        print(f"[Report] Session {session_id} not found.")
        return

    records = get_attendance_for_session(conn, session_id)

    print("\n" + "=" * 60)
    print(f"  ATTENDANCE REPORT")
    print(f"  Class: {session.get('class_name', 'N/A')}")
    print(f"  Session: {session_id[:8]}...")
    print(f"  Start: {session.get('start_time', 'N/A')}")
    print(f"  End:   {session.get('end_time', 'N/A')}")
    print("=" * 60)

    if not records:
        print("  No attendance records.")
    else:
        present_count = sum(1 for r in records if r["status"] == "Present")
        late_count = sum(1 for r in records if r["status"] == "Late")

        print(f"  {'Student ID':<15} {'Name':<20} {'Status':<10} {'First Seen':<20}")
        print(f"  {'-'*15} {'-'*20} {'-'*10} {'-'*20}")
        for rec in records:
            first = rec.get("first_seen", "")[:19]
            print(f"  {rec['student_id']:<15} {rec['name']:<20} {rec['status']:<10} {first:<20}")

        print(f"\n  Total: {len(records)} | Present: {present_count} | Late: {late_count}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export attendance report")
    parser.add_argument("--session_id", help="Session ID (leave empty to list all sessions)")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    args = parser.parse_args()

    conn = init_db()

    if args.list or args.session_id is None:
        sessions = get_all_sessions(conn)
        if not sessions:
            print("[Report] No sessions found.")
        else:
            print(f"\n{'Session ID':<40} {'Class':<15} {'Start':<20} {'End':<20}")
            print(f"{'-'*40} {'-'*15} {'-'*20} {'-'*20}")
            for s in sessions:
                sid = s["session_id"][:36]
                cls = s.get("class_name", "N/A") or "N/A"
                start = (s.get("start_time") or "")[:19]
                end = (s.get("end_time") or "ongoing")[:19]
                print(f"{sid:<40} {cls:<15} {start:<20} {end:<20}")
            print()

        if args.session_id is None:
            conn.close()
            return

    print_summary(conn, args.session_id)
    csv_path = export_report(conn, args.session_id, args.out)
    if csv_path:
        print(f"[Report] CSV saved to: {csv_path}")

    conn.close()


if __name__ == "__main__":
    main()
