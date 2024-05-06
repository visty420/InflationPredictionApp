from sqlalchemy import select
from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_user(db: AsyncSession, user_id: int):
    return await db.execute(select(models.User).filter(models.User.id == user_id)).scalars().first()


async def create_user(db: AsyncSession, user_data: schemas.UserCreate):
    db_user = models.User(
        username=user_data.username,
        hashed_password=pwd_context.hash(user_data.password),
        email=user_data.email
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
