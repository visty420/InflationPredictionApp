import csv
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jwt
from . import crud, models, schemas
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession 
from passlib.context import CryptContext
from .crud import pwd_context
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="Frontend"), name="static")
templates = Jinja2Templates(directory="Frontend/templates")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


PASSWORD_REGEX = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("mainpage.html", {"request": request})

@app.get("/factors")
def macroeconomicdata_page(request : Request):
    return templates.TemplateResponse("macroeconomic_data.html", {"request" : request})

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(request: Request, username: str = Form(...), password: str = Form(...), email: str = Form(...), db: AsyncSession = Depends(models.get_db)):
            # Validate email
    if not re.match(EMAIL_REGEX, email):
        print("Email validation failed")  
        return HTMLResponse(content="<script>alert('Invalid email format'); window.location='/register';</script>")
    
    # Validate password
    if not re.match(PASSWORD_REGEX, password):
        print("Password validation failed")  
        return HTMLResponse(content="<script>alert('Password must contain at least 8 characters, including an uppercase letter, a symbol, and a digit'); window.location='/register';</script>")

    # Proceed with user creation if validations pass
    user_data = schemas.UserCreate(username=username, email=email, password=password)
    new_user = await crud.create_user(db, user_data)
    print("User created successfully")
    return HTMLResponse(content="<script>alert('User registered successfully!'); window.location='/login';</script>")



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)



def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/login")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(models.get_db)):
    user = await crud.get_user(form_data.username, db)
    if not user or not verify_password(form_data.password, user.hashed_password):
        return HTMLResponse(content="<script>alert('Invalid credentials'); window.location='/login';</script>")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    response_content = "<script>alert('User logged in successfully!'); window.location='/';</script>"
    response = HTMLResponse(content=response_content)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.post("/insert_data")
async def insert_data(date: str = Form(...), cpi: str = Form(...), ppi: str = Form(...),
                      pce: str = Form(...), fedfunds: str = Form(...), unrate: str = Form(...),
                      gdp: str = Form(...), m2sl: str = Form(...), umcsent: str = Form(...),
                      wagegrowth: str = Form(...), inflrate: str = Form(...)):
    data = [date, cpi, ppi, pce, fedfunds, unrate, gdp, m2sl, umcsent, wagegrowth, inflrate]
    try:
        with open('./Backend/Data/complete_data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        return HTMLResponse(content="<script>alert('Data inserted successfully!'); window.location.href='/factors';</script>")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")
    
@app.get("/view_data")
async def view_data():
    import pandas as pd
    df = pd.read_csv("./Backend/Data/complete_data.csv")
    return HTMLResponse(content=df.to_html(classes="data", border=0))