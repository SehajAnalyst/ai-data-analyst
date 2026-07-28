"""
db/repository/session_factory.py
================================

Creates the SQLAlchemy engine and Session factory for the application's
internal SQLite database.

This is completely separate from db/connectors/, which manages the
user's database (SQLite/MySQL/PostgreSQL) that the AI queries.
"""

from pathlib import Path
from db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_PATH = Path("data/app_internal.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


def get_session():
    """
    Return a new SQLAlchemy session.
    """
    return SessionLocal()


def initialize_database():
    """
    Create all internal tables.
    """
    Base.metadata.create_all(engine)
    
    