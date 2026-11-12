import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo(os.environ.get("MEMORY_TZ", "Asia/Shanghai"))
WEEK = "一二三四五六日"

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, conv TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL  -- ISO，带时区
);
CREATE TABLE IF NOT EXISTS capsules (
  id INTEGER PRIMARY KEY, conv TEXT NOT NULL, last_message_id INTEGER NOT NULL,
  start_at TEXT NOT NULL, end_at TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS day_summaries (
  conv TEXT, day TEXT, seg INTEGER, text TEXT NOT NULL,
  first_id INTEGER NOT NULL, last_id INTEGER NOT NULL,  -- 这份总结是从哪几条原话写的
  PRIMARY KEY (conv, day, seg)
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY, conv TEXT, day TEXT, seg INTEGER,
  text TEXT NOT NULL, vec BLOB NOT NULL
);
"""


def connect(path=":memory:"):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def add_message(db, conv, role, content, created_at=None):
    created_at = (created_at or datetime.now(TZ)).isoformat()
    cur = db.execute(
        "INSERT INTO messages (conv, role, content, created_at) VALUES (?,?,?,?)",
        (conv, role, content, created_at),
    )
    db.commit()
    return cur.lastrowid


def local(iso):
    return datetime.fromisoformat(iso).astimezone(TZ)


def stamp(iso):
    """[11-07 周六 20:58]。只由消息自己的时间决定，同一段历史每次拼出来一字不差。"""
    t = local(iso)
    return f"[{t:%m-%d} 周{WEEK[t.weekday()]} {t:%H:%M}]"
