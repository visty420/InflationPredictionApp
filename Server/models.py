from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

SQLALCHEMY_DATABASE_URL = "postgresql+asyncpg://admin1:admin@localhost:5432/ThesisDB"
engine = create_async_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)


async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    async with async_session() as session:
        yield session
