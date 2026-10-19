"""Database utilities for the data manager."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine,event
from sqlalchemy.orm import sessionmaker

from .models import Base

# Bump together with a new entry in ``_MIGRATIONS``. Migrations are frozen SQL:
# never edit one that has shipped, add a new step instead.
SCHEMA_VERSION = 1


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _execute_all(conn: sqlite3.Connection, script: str) -> None:
    """Run statements inside the caller's transaction (``executescript`` would commit)."""
    for statement in script.split(";"):
        if statement.strip():
            conn.execute(statement)


def _migrate_1_model_snapshots(conn: sqlite3.Connection) -> None:
    """Make training metrics optional and add snapshot columns to model versions.

    SQLite cannot relax NOT NULL in place, so the table is rebuilt following the
    documented create-copy-drop-rename procedure (foreign keys are off).
    """
    columns = _columns(conn, "model_versions")
    if not columns or "content_hash" in columns:
        return
    _execute_all(
        conn,
        """
        CREATE TABLE model_versions_new (
            id INTEGER NOT NULL,
            name VARCHAR(80) NOT NULL,
            project_id INTEGER NOT NULL,
            model_type VARCHAR(20) NOT NULL,
            data_size INTEGER,
            energy FLOAT,
            force FLOAT,
            virial FLOAT,
            model_path TEXT NOT NULL,
            notes TEXT NOT NULL,
            parent_id INTEGER,
            created_at DATETIME NOT NULL,
            content_hash VARCHAR(64),
            model_kind VARCHAR(40) NOT NULL,
            elements JSON NOT NULL,
            source_path TEXT NOT NULL,
            dataset_ref JSON NOT NULL,
            starred BOOLEAN NOT NULL,
            auto_captured BOOLEAN NOT NULL,
            last_seen_at DATETIME NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE,
            FOREIGN KEY(parent_id) REFERENCES model_versions (id) ON DELETE CASCADE
        );
        INSERT INTO model_versions_new (
            id, name, project_id, model_type, data_size, energy, force, virial,
            model_path, notes, parent_id, created_at, content_hash, model_kind,
            elements, source_path, dataset_ref, starred, auto_captured, last_seen_at
        )
        SELECT id, name, project_id, model_type, data_size, energy, force, virial,
               model_path, notes, parent_id, created_at, NULL, '',
               '[]', '', '{}', 0, 0, created_at
        FROM model_versions;
        DROP TABLE model_versions;
        ALTER TABLE model_versions_new RENAME TO model_versions;
        CREATE INDEX ix_model_versions_project_id ON model_versions (project_id);
        CREATE INDEX ix_model_versions_parent_id ON model_versions (parent_id);
        CREATE INDEX ix_model_versions_content_hash ON model_versions (content_hash);
        """
    )


_MIGRATIONS = {1: _migrate_1_model_snapshots}


def migrate(path: str | Path) -> None:
    """Upgrade an existing database file to ``SCHEMA_VERSION``.

    A brand-new file is created directly at the latest schema by
    ``create_all``, so only files that already exist are migrated.
    """
    path = Path(path)
    if not path.exists():
        return
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        conn.execute("PRAGMA foreign_keys=OFF")
        for step in range(version + 1, SCHEMA_VERSION + 1):
            conn.execute("BEGIN")
            try:
                _MIGRATIONS[step](conn)
                problems = conn.execute("PRAGMA foreign_key_check").fetchall()
                if problems:
                    raise RuntimeError(f"Database migration {step} broke foreign keys: {problems[:3]}")
                conn.execute(f"PRAGMA user_version={step}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.close()


class Database:
    """Wraps an SQLAlchemy engine and session factory."""

    def __init__(self, path: str | Path = "mlpman.db") -> None:
        self.path = Path(path)
        self.first=not self.path.exists()
        migrate(self.path)
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)
        @event.listens_for(self.engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON;")
            # Let the GUI and the command line read while the other writes.
            dbapi_conn.execute("PRAGMA busy_timeout=5000;")
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def session(self):
        """Create a new SQLAlchemy session."""
        return self.Session()
