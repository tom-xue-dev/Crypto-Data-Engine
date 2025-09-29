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
    # Import models to ensure metadata is populated before create_all
    from crypto_data_engine.db.models.download import DownloadTask  # noqa: F401
    from crypto_data_engine.db.models.aggregate import AggregateTask  # noqa: F401
    from crypto_data_engine.db.models.Base import Base
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 as test")).scalar_one()
        db_info = conn.execute(text("SELECT current_database(), current_user")).fetchone()
        print("✅ Database connection succeeded!")
        print(f"   Test query result: {result}")
        print(f"   Current database: {db_info[0]}")
        print(f"   Current user: {db_info[1]}")


if __name__ == "__main__":
    pass
