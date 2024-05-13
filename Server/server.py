import csv
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import jwt
import numpy as np
import torch
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

# # Load models and scalers
# arima_model = joblib.load('./SavedModels/arima_model.pkl')
# lstm_model = torch.load('./SavedModels/lstm_model.pth')
# nn_3_model = torch.load('./SavedModels/3in_mlp.pth')
# nn_9_model = torch.load('./SavedModels/9in_mlp.pth')
# rnn_model = torch.load('./SavedModels/rnn.pth')

# # Assume scalers are stored as .gz files
# nn_3_scaler = joblib.load('./SavedModels/3in_mlp_scaler.gz')
# nn_9_scaler = joblib.load('./SavedModels/9in_mlp_scaler.gz')
# lstm_scaler = joblib.load('./SavedModels/lstm_scaler.gz')

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

@app.get("/predictions")
def predictions_page(request : Request):
    return templates.TemplateResponse("predictions.html", {"request" : request})

@app.post("/register")
async def register_user(request: Request, username: str = Form(...), password: str = Form(...), email: str = Form(...), db: AsyncSession = Depends(models.get_db)):
    if not re.match(EMAIL_REGEX, email):
        print("Email validation failed")  
        return HTMLResponse(content="<script>alert('Invalid email format'); window.location='/register';</script>")
    if not re.match(PASSWORD_REGEX, password):
        print("Password validation failed")  
        return HTMLResponse(content="<script>alert('Password must contain at least 8 characters, including an uppercase letter, a symbol, and a digit'); window.location='/register';</script>")
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

# def predict_with_arima(steps: int):
#     # Assuming steps is a parameter used to predict future values
#     prediction = arima_model.forecast(steps=steps)
#     return prediction.tolist()

# def predict_with_lstm(features):
#     features = np.array([features]).astype(float)
#     features = lstm_scaler.transform(features)
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     lstm_model.eval()
#     with torch.no_grad():
#         predicted = lstm_model(features_tensor).item()
#     return predicted

# def predict_with_nn_3(features):
#     features = np.array([features]).astype(float)
#     features = nn_3_scaler.transform(features)
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     nn_3_model.eval()
#     with torch.no_grad():
#         predicted = nn_3_model(features_tensor).item()
#     return predicted

# def predict_with_nn_9(features):
#     features = np.array([features]).astype(float)
#     features = nn_9_scaler.transform(features)
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     nn_9_model.eval()
#     with torch.no_grad():
#         predicted = nn_9_model(features_tensor).item()
#     return predicted

# def predict_with_rnn(features):
#     features = np.array([features]).astype(float)
#     features = nn_9_scaler.transform(features)  # Assuming RNN uses the same scaler as NN_9
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     rnn_model.eval()
#     with torch.no_grad():
#         predicted = rnn_model(features_tensor).item()
#     return predicted

# @router.post("/predict/")
# async def make_prediction(data: dict, model_type: str, username: str = Depends(oauth2_scheme)):
#     features = data.get('features', [])
#     steps = int(data.get('steps', 1))  
#     if model_type == 'ARIMA':
#         result = predict_with_arima(steps)
#     elif model_type == 'LSTM':
#         result = predict_with_lstm(features)
#     elif model_type == 'NN_3':
#         result = predict_with_nn_3(features[:3])  
#     elif model_type == 'NN_9':
#         result = predict_with_nn_9(features)
#     elif model_type == 'RNN':
#         result = predict_with_rnn(features)
#     else:
#         raise HTTPException(status_code=400, detail="Model type not supported")
#     return {"predicted_inflation": result}