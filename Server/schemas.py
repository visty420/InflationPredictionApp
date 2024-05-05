from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int

    class Config:
        orm_mode = True

from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
