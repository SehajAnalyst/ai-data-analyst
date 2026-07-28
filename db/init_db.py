"""
db/init_db.py
=============

Creates the application's internal database from SQLAlchemy models.
Run this once whenever the schema changes during development.
"""

from db.models import Base
from db.repository.session_factory import engine


def create_database() -> None:
    Base.metadata.create_all(bind=engine)
    print("✅ Internal database created successfully.")


if __name__ == "__main__":
    create_database()