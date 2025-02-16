from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import conn_str

# Создаем движок SQLAlchemy
Base = declarative_base()
engine = create_engine(conn_str)
SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


# Создаем таблицы в БД
def init_db():
    Base.metadata.create_all(engine)
    print("✅ База данных и таблицы успешно созданы!")


if __name__ == "__main__":
    init_db()
