from typing import List
from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    model_name: str
    features: List[float]

class ARIMAPredictionRequest(BaseModel):
    months: int  

class ComparisonRequest(BaseModel):
    features: List[float]

class ModelRequest(BaseModel):
    model_name: str