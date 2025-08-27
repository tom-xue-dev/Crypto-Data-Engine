from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from crypto_data_engine.common.config.config_settings import settings


cfg = settings.db_cfg
engine = create_engine(
    cfg.db_url,
    pool_pre_ping=True,
    pool_size=cfg.db_pool_size,
    pool_timeout=cfg.db_pool_timeout,
    pool_recycle=cfg.db_pool_recycle,
    echo=cfg.db_echo
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
def init():
    from crypto_data_engine.db.models.download import DownloadTask
    from crypto_data_engine.db.models.Base import Base
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 as test")).scalar_one()
        db_info = conn.execute(text("SELECT current_database(), current_user")).fetchone()
        print(f"✅ 数据库连接成功!")
        print(f"   测试查询结果: {result}")
        print(f"   当前数据库: {db_info[0]}")
        print(f"   当前用户: {db_info[1]}")

if __name__ == "__main__":
  pass