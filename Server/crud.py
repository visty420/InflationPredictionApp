from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
from .database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
async def create_user(db: AsyncSession, user: schemas.UserCreate) -> models.User:
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=pwd_context.hash(user.password)
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user