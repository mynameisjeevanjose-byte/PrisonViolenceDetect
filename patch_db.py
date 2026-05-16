import sqlite3
import os

basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'prison_system.db')

def patch_db():
    print(f"Patching database at: {db_path}")
    if not os.path.exists(db_path):
        print("Error: Database file not found. Run reset_db.py first.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Add rejection_reason column
    try:
        c.execute("ALTER TABLE inmate ADD COLUMN rejection_reason VARCHAR(255)")
        print("[SUCCESS] Added column: rejection_reason")
    except sqlite3.OperationalError as e:
        print(f"[INFO] rejection_reason: {e}")

    # 2. Add room_number column
    try:
        c.execute("ALTER TABLE inmate ADD COLUMN room_number INTEGER DEFAULT 0")
        print("[SUCCESS] Added column: room_number")
    except sqlite3.OperationalError as e:
        print(f"[INFO] room_number: {e}")

    # 3. Add release_date column (For History Feature)
    try:
        c.execute("ALTER TABLE inmate ADD COLUMN release_date DATETIME")
        print("[SUCCESS] Added column: release_date")
    except sqlite3.OperationalError as e:
        print(f"[INFO] release_date: {e}")

    # 4. Add dob column
    try:
        c.execute("ALTER TABLE inmate ADD COLUMN dob DATE")
        print("[SUCCESS] Added column: dob")
    except sqlite3.OperationalError as e:
        print(f"[INFO] dob: {e}")

    # 5. Add address column
    try:
        c.execute("ALTER TABLE inmate ADD COLUMN address TEXT")
        print("[SUCCESS] Added column: address")
    except sqlite3.OperationalError as e:
        print(f"[INFO] address: {e}")

    conn.commit()
    conn.close()
    print("Database patch complete.")

if __name__ == '__main__':
    patch_db()