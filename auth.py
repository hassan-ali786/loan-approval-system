from flask_login import UserMixin
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash


# ─────────────────────────────────────────
# User model for Flask-Login
# ─────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id       = id
        self.username = username
        self.email    = email


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
            created_at    TEXT
        )
    ''')
    # Add user_id column to predictions if missing
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER DEFAULT NULL")
    except Exception:
        pass  # Column already exists
    conn.commit()
    conn.close()


def create_user(username, email, password):
    """Register a new user. Returns True on success, False if duplicate."""
    try:
        conn = sqlite3.connect("history.db")
        c    = conn.cursor()
        c.execute('''
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, datetime('now'))
        ''', (username, email, generate_password_hash(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # username or email already taken


def get_user_by_email(email):
    """Fetch user row by email."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT id, username, email, password_hash FROM users WHERE email = ?", (email,))
    row  = c.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    """Fetch user row by id (used by Flask-Login user_loader)."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    row  = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2])
    return None


def verify_password(stored_hash, password):
    """Check plaintext password against stored hash."""
    return check_password_hash(stored_hash, password)