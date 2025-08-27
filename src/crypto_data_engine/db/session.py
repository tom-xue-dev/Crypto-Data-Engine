from sqlalchemy.orm import sessionmaker
from crypto_data_engine.db.db_init import engine
import functools
from typing import Callable, Any

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def with_db_session(func: Callable) -> Callable:
    """
    自动注入数据库会话装饰器
    - 如果参数中已有db参数，则使用传入的session
    - 如果没有db参数，则自动创建和管理session
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查是否已经有db参数
        if 'db' in kwargs:
            # 直接使用传入的session
            return func(*args, **kwargs)

        # 自动创建session
        db = SessionLocal()
        try:
            # 将db注入到kwargs中
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
    数据库事务装饰器
    - 自动管理事务的提交和回滚
    - 出错时自动回滚
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查是否已经有db参数
        if 'db' in kwargs:
            # 使用传入的session，不管理事务
            return func(*args, **kwargs)

        # 自动创建session和管理事务
        db = SessionLocal()
        try:
            kwargs['db'] = db
            result = func(*args, **kwargs)
            db.commit()  # 成功时提交事务
            return result
        except Exception as e:
            db.rollback()  # 出错时回滚
            raise e
        finally:
            db.close()

    return wrapper



