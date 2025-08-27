"""Database utilities for the data manager."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


class Database:
    """Wraps an SQLAlchemy engine and session factory."""

    def __init__(self, path: str | Path = "mlpman.db") -> None:
        self.path = Path(path)
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def session(self):
        """Create a new SQLAlchemy session."""
        return self.Session()