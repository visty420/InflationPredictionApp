from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def create_user(user: schemas.UserCreate):
    db = models.SessionLocal()
    db_user = models.User(
        username=user.username,
        hashed_password=pwd_context.hash(user.password),
        email=user.email
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    db.close()
    return db_user