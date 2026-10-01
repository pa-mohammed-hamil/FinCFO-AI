"""
Database Connection Module

Provides SQLAlchemy base and engine setup.
"""

from sqlalchemy.ext.declarative import declarative_base

# SQLAlchemy declarative base for models
Base = declarative_base()

__all__ = ["Base"]
