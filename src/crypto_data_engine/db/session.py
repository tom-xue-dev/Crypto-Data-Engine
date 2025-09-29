from sqlalchemy.orm import sessionmaker
from crypto_data_engine.db.db_init import engine
import functools
from typing import Callable, Any

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def with_db_session(func: Callable) -> Callable:
    """
    Decorator that injects a database session.
    - Use provided `db` argument if supplied.
    - Otherwise create, inject, and manage the session automatically.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if session already provided
        if 'db' in kwargs:
            # Use the supplied session
            return func(*args, **kwargs)

        # Create session automatically
        db = SessionLocal()
        try:
            # Inject session into kwargs
            kwargs['db'] = db
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    return wrapper


def with_db_transaction(func: Callable) -> Callable:
    """
    Transactional decorator.
    - Automatically manage commit/rollback.
    - Roll back on error.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if session already provided
        if 'db' in kwargs:
            # Use provided session; caller handles transaction
            return func(*args, **kwargs)

        # Create session and manage transaction automatically
        db = SessionLocal()
        try:
            kwargs['db'] = db
            result = func(*args, **kwargs)
            db.commit()  # Commit on success
            return result
        except Exception as e:
            db.rollback()  # Roll back on failure
            raise e
        finally:
            db.close()

    return wrapper



