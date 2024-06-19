from sqlalchemy import Column, Date, Integer, Numeric, String, create_engine
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

class MacroeconomicData(Base):
    __tablename__ = "macroeconomic_data"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    cpi = Column(Numeric(10, 2), nullable=False)
    ppi = Column(Numeric(10, 2), nullable=False)
    pce = Column(Numeric(10, 2), nullable=False)
    fedfunds = Column(Numeric(10, 2), nullable=False)
    unrate = Column(Numeric(10, 2), nullable=False)
    gdp = Column(Numeric(10, 2), nullable=False)
    m2sl = Column(Numeric(10, 2), nullable=False)
    umcsent = Column(Numeric(10, 2), nullable=False)
    wagegrowth = Column(Numeric(10, 2), nullable=False)
    inflrate = Column(Numeric(10, 2), nullable=False)

async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    async with async_session() as session:
        yield session

