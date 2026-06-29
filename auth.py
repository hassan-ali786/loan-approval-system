from flask_login import UserMixin
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash


# ─────────────────────────────────────────
# User model for Flask-Login
# ─────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, email, is_admin=0):
        self.id       = id
        self.username = username
        self.email    = email
        self.is_admin = bool(is_admin)


# ─────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────
def init_user_db():
    """Create users table if it doesn't exist."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin      INTEGER DEFAULT 0,
            is_blocked    INTEGER DEFAULT 0,
            created_at    TEXT
        )
    ''')
    # Backfill columns for DBs created before this update
    for col, ddl in [
        ("is_admin",   "ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0"),
        ("is_blocked", "ALTER TABLE users ADD COLUMN is_blocked INTEGER DEFAULT 0"),
    ]:
        try:
            c.execute(ddl)
        except Exception:
            pass  # Column already exists

    # user_id link on predictions table
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER DEFAULT NULL")
    except Exception:
        pass

    conn.commit()
    conn.close()


def create_user(username, email, password):
    """
    Register a new user. Returns True on success, False if duplicate.
    The very first account created in the system is automatically promoted to admin.
    """
    try:
        conn = sqlite3.connect("history.db")
        c    = conn.cursor()

        c.execute("SELECT COUNT(*) FROM users")
        is_first_user = c.fetchone()[0] == 0
        admin_flag    = 1 if is_first_user else 0

        c.execute('''
            INSERT INTO users (username, email, password_hash, is_admin, is_blocked, created_at)
            VALUES (?, ?, ?, ?, 0, datetime('now'))
        ''', (username, email, generate_password_hash(password), admin_flag))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # username or email already taken


def get_user_by_email(email):
    """Fetch user row by email — includes is_admin, is_blocked, password_hash."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("""
        SELECT id, username, email, password_hash, is_admin, is_blocked
        FROM users WHERE email = ?
    """, (email,))
    row = c.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    """Fetch user row by id (used by Flask-Login user_loader)."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT id, username, email, is_admin FROM users WHERE id = ?", (user_id,))
    row  = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2], row[3])
    return None


def verify_password(stored_hash, password):
    """Check plaintext password against stored hash."""
    return check_password_hash(stored_hash, password)


def get_password_hash(user_id):
    """Fetch a user's current password hash — used to verify current password before a change."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
    row  = c.fetchone()
    conn.close()
    return row[0] if row else None


def update_password(user_id, new_password):
    """Set a new password hash for a user."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("UPDATE users SET password_hash = ? WHERE id = ?",
              (generate_password_hash(new_password), user_id))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# Forgot-password reset tokens
# ─────────────────────────────────────────
def init_reset_table():
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reset_tokens (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT,
            used       INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def create_reset_token(user_id):
    """Generate and store a one-time password reset token. Returns the token string."""
    import secrets
    token = secrets.token_urlsafe(32)
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute('''
        INSERT INTO reset_tokens (token, user_id, created_at, used)
        VALUES (?, ?, datetime('now'), 0)
    ''', (token, user_id))
    conn.commit()
    conn.close()
    return token


def get_reset_token(token):
    """Look up a reset token. Returns (user_id, created_at, used) or None."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT user_id, created_at, used FROM reset_tokens WHERE token = ?", (token,))
    row = c.fetchone()
    conn.close()
    return row


def mark_token_used(token):
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("UPDATE reset_tokens SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# Admin helpers
# ─────────────────────────────────────────
def get_all_users():
    """Return every user with their prediction count, for the admin panel."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute('''
        SELECT u.id, u.username, u.email, u.is_admin, u.is_blocked, u.created_at,
               COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        GROUP BY u.id
        ORDER BY u.id ASC
    ''')
    rows = c.fetchall()
    conn.close()
    return rows


def toggle_block_user(user_id):
    """Flip a user's blocked status. Returns the new status (1=blocked, 0=active)."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT is_blocked FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    new_status = 0 if row[0] else 1
    c.execute("UPDATE users SET is_blocked = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return new_status


def delete_user(user_id):
    """Delete a user and all their predictions. Admins cannot be deleted via this helper."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row or row[0] == 1:
        conn.close()
        return False  # don't allow deleting admins
    c.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True


def is_user_blocked(user_id):
    """Quick check used at login time."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT is_blocked FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return bool(row[0]) if row else False