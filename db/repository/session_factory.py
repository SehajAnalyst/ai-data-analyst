"""
db/repository/session_factory.py
================================

Creates the SQLAlchemy engine and Session factory for the application's
internal database.

This database stores application state such as conversations and
query history. It is separate from the user's queryable database.
"""

from config.settings import get_settings
from db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


settings = get_settings()

DATABASE_URL = settings.internal_db_url


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