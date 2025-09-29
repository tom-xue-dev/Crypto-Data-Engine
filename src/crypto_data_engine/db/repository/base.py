"""
Base repository class with session-aware decorators.
"""
from typing import Type, TypeVar, Generic, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from crypto_data_engine.db.session import with_db_transaction, with_db_session


ModelType = TypeVar("ModelType")

class BaseRepository(Generic[ModelType]):
    """Base repository providing common CRUD operations."""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    @classmethod
    @with_db_transaction
    def create(cls, *args, **kwargs) -> ModelType:
        """Create record."""
        db: Session = kwargs.pop('db')  # Injected Session instance
        instance = cls._get_model()(**kwargs)
        db.add(instance)
        db.flush()  # Populate ID without committing
        db.refresh(instance)  # Ensure all attributes are loaded

        # Critical step: detach instance from session while keeping values
        db.expunge(instance)

        return instance


    @classmethod
    @with_db_session
    def get_by_id(cls, record_id: int, **kwargs) -> Optional[ModelType]:
        """Fetch record by ID."""
        db: Session = kwargs.pop('db')
        return db.query(cls._get_model()).filter(cls._get_model().id == record_id).first()

    @classmethod
    @with_db_session
    def get_by_kwargs(cls, **kwargs) -> Optional[ModelType]:
        """Fetch single record via filters."""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)
        return query.first()

    @classmethod
    @with_db_session
    def get_all(
            cls,
            offset: int = None,
            limit: int = None,
            order_by: str = "id",
            desc_order: bool = False,
            **kwargs
    ) -> List[ModelType]:
        """Retrieve records list."""
        db: Session = kwargs.pop("db")
        model = cls._get_model()
        query = db.query(model)
        for key, value in kwargs.items():
            if hasattr(model, key):
                col = getattr(model, key)
                if isinstance(value, (list, tuple)):
                    if value:  # Avoid empty IN ()
                        query = query.filter(col.in_(value))
                    else:
                        return []
                else:
                    query = query.filter(col == value)
        if hasattr(model, order_by):
            order_column = getattr(model, order_by)
            query = query.order_by(desc(order_column) if desc_order else asc(order_column))
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return query.all()

    @classmethod
    @with_db_session
    def count(cls, **kwargs) -> int:
        """Count records."""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        # Apply filters
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                if isinstance(value, (list, tuple)):
                    query = query.filter(getattr(cls._get_model(), key).in_(value))
                else:
                    query = query.filter(getattr(cls._get_model(), key) == value)

        return query.count()

    @classmethod
    @with_db_transaction
    def update(cls, record_id: int, **kwargs) -> Optional[ModelType]:
        """Update record."""
        db: Session = kwargs.pop('db')
        instance = db.query(cls._get_model()).filter(cls._get_model().id == record_id).first()
        if not instance:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        db.flush()
        db.refresh(instance)
        return instance

    @classmethod
    @with_db_transaction
    def update_by_kwargs(cls, filter_kwargs: Dict[str, Any], **update_kwargs) -> List[ModelType]:
        """Bulk update by filters."""
        db: Session = update_kwargs.pop('db')
        query = db.query(cls._get_model())

        # Build filter conditions
        for key, value in filter_kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        instances = query.all()
        for instance in instances:
            for key, value in update_kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
        db.flush()
        for instance in instances:
            db.refresh(instance)
        return instances

    @classmethod
    @with_db_transaction
    def delete(cls, record_id: int, **kwargs) -> bool:
        """Delete record."""
        db: Session = kwargs.pop('db')

        instance = db.query(cls._get_model()).filter(cls._get_model().id == record_id).first()
        if not instance:
            return False

        db.delete(instance)
        return True

    @classmethod
    @with_db_transaction
    def delete_by_kwargs(cls, **kwargs) -> int:
        """Bulk delete by filters."""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        # Build filter conditions
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        deleted_count = query.count()
        query.delete(synchronize_session=False)
        return deleted_count

    @classmethod
    @with_db_transaction
    def bulk_create(cls, data_list: List[Dict[str, Any]], **kwargs) -> List[ModelType]:
        """Bulk create records."""
        db: Session = kwargs.pop('db')

        instances = []
        for data in data_list:
            instance = cls._get_model()(**data)
            instances.append(instance)

        db.add_all(instances)
        db.flush()
        for instance in instances:
            db.refresh(instance)
        return instances

    @classmethod
    @with_db_session
    def exists(cls, **kwargs) -> bool:
        """Check record existence."""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        return query.first() is not None

    @classmethod
    @with_db_transaction
    def get_or_create(cls, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[ModelType, bool]:
        """Get or create record, returning (instance, created_flag)."""
        db: Session = kwargs.pop('db')

        # Attempt to fetch first
        query = db.query(cls._get_model())
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        instance = query.first()
        if instance:
            return instance, False

        # Create new record
        create_kwargs = {**kwargs}
        if defaults:
            create_kwargs.update(defaults)

        instance = cls._get_model()(**create_kwargs)
        db.add(instance)
        db.flush()
        db.refresh(instance)
        return instance, True

    @classmethod
    def _get_model(cls) -> Type[ModelType]:
        """Return model class – subclasses must override or define `_model`."""
        if not hasattr(cls, '_model'):
            raise NotImplementedError("Subclasses must define `_model` attribute or override `_get_model`.")
        return cls._model