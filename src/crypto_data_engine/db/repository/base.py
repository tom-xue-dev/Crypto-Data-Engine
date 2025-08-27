"""
基础Repository类 - 使用装饰器注入session
"""
from typing import Type, TypeVar, Generic, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from crypto_data_engine.db.session import with_db_transaction, with_db_session


ModelType = TypeVar("ModelType")

class BaseRepository(Generic[ModelType]):
    """基础Repository类，提供通用的CRUD操作"""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    @classmethod
    @with_db_transaction
    def create(cls, **kwargs) -> ModelType:
        """创建记录"""
        db: Session = kwargs.pop('db')  # 从装饰器注入的db
        instance = cls._get_model()(**kwargs)
        db.add(instance)
        db.flush()  # 获取ID但不提交
        db.refresh(instance)  # 确保所有属性都被加载

        # 🔥 关键修复：让对象脱离 session，但保持属性值
        db.expunge(instance)

        return instance


    @classmethod
    @with_db_session
    def get_by_id(cls, record_id: int, **kwargs) -> Optional[ModelType]:
        """根据ID获取记录"""
        db: Session = kwargs.pop('db')
        return db.query(cls._get_model()).filter(cls._get_model().id == record_id).first()

    @classmethod
    @with_db_session
    def get_by_kwargs(cls, **kwargs) -> Optional[ModelType]:
        """根据关键字参数获取单条记录"""
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
        offset: int = 0,
        limit: int = 100,
        order_by: str = "id",
        desc_order: bool = False,
        **kwargs
    ) -> List[ModelType]:
        """获取记录列表"""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        # 应用过滤条件
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                if isinstance(value, (list, tuple)):
                    # 支持IN查询
                    query = query.filter(getattr(cls._get_model(), key).in_(value))
                else:
                    query = query.filter(getattr(cls._get_model(), key) == value)

        # 排序
        if hasattr(cls._get_model(), order_by):
            order_column = getattr(cls._get_model(), order_by)
            if desc_order:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(asc(order_column))

        return query.offset(offset).limit(limit).all()

    @classmethod
    @with_db_session
    def count(cls, **kwargs) -> int:
        """统计记录数量"""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        # 应用过滤条件
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
        """更新记录"""
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
        """根据条件批量更新"""
        db: Session = update_kwargs.pop('db')
        query = db.query(cls._get_model())

        # 构建过滤条件
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
        """删除记录"""
        db: Session = kwargs.pop('db')

        instance = db.query(cls._get_model()).filter(cls._get_model().id == record_id).first()
        if not instance:
            return False

        db.delete(instance)
        return True

    @classmethod
    @with_db_transaction
    def delete_by_kwargs(cls, **kwargs) -> int:
        """根据条件批量删除"""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        # 构建过滤条件
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        deleted_count = query.count()
        query.delete(synchronize_session=False)
        return deleted_count

    @classmethod
    @with_db_transaction
    def bulk_create(cls, data_list: List[Dict[str, Any]], **kwargs) -> List[ModelType]:
        """批量创建记录"""
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
        """检查记录是否存在"""
        db: Session = kwargs.pop('db')
        query = db.query(cls._get_model())

        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        return query.first() is not None

    @classmethod
    @with_db_transaction
    def get_or_create(cls, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[ModelType, bool]:
        """获取或创建记录，返回(实例, 是否新创建)"""
        db: Session = kwargs.pop('db')

        # 先尝试获取
        query = db.query(cls._get_model())
        for key, value in kwargs.items():
            if hasattr(cls._get_model(), key):
                query = query.filter(getattr(cls._get_model(), key) == value)

        instance = query.first()
        if instance:
            return instance, False

        # 创建新记录
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
        """获取模型类 - 子类需要重写此方法"""
        if not hasattr(cls, '_model'):
            raise NotImplementedError("子类必须定义 _model 属性或重写 _get_model 方法")
        return cls._model