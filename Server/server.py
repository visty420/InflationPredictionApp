from datetime import datetime, timedelta
import logging
import os
import aiosmtplib
from fastapi import FastAPI, Request, Form, HTTPException, Depends, logger, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError
import pandas as pd
from sqlalchemy import select
from . import crud, models, schemas
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession 
from passlib.context import CryptContext
from .crud import pwd_context
from torch import nn
import re
from .schemas import ARIMAPredictionRequest, ComparisonRequest, PredictionRequest
import joblib
import jwt
import numpy as np
import torch
import csv
from .models import User, get_db
from .auth import authenticate_user, oauth2_scheme, verify_token, SECRET_KEY,ALGORITHM
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from sqlalchemy.orm import Session

app = FastAPI()
app.mount("/static", StaticFiles(directory="Frontend"), name="static")
app.mount("/auxiliaries", StaticFiles(directory="Auxiliaries"), name="auxiliaries")
templates = Jinja2Templates(directory="Frontend/templates")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
PASSWORD_REGEX = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class InflationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(InflationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  
        return out

def custom_load(path):
    return torch.load(path, map_location=lambda storage, loc: storage)

rnn_scaler_path = "./Backend/SavedModels/rnn_scaler.gz"
rnn_model_path = "./Backend/SavedModels/rnn_model.pth"

try:
    rnn_scaler = joblib.load(rnn_scaler_path)
    rnn_model = InflationRNN(input_size=9, hidden_size=50, num_layers=2, output_size=1)
    rnn_model.load_state_dict(custom_load(rnn_model_path))
    rnn_model.eval()
    print("RNN model and scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load RNN model or scaler: {e}")

@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("mainpage.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("Server/favicon.ico")

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

@app.get("/models")
def comparison_page(request : Request):
    return templates.TemplateResponse("comparison.html", {"request": request})

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
    
@app.get("/recent_data")
async def recent_data():
    csv_file_path = './Backend/Data/complete_data.csv'
    data = pd.read_csv(csv_file_path).tail(15)
    return data.to_dict(orient='records')

@app.get("/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        return JSONResponse(status_code=400, content={"detail": "Email already registered"})
    return JSONResponse(status_code=200, content={"detail": "Email available"})


arima_model = joblib.load("./Backend/SavedModels/arima_model_212.pkl")
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

lstm_scaler = joblib.load("./Backend/SavedModels/lstmscaler.gz")
lstm_model = LSTMModel(input_dim=9, hidden_dim=31, num_layers=2, output_dim=1)
lstm_model.load_state_dict(torch.load("./Backend/SavedModels/lstmmodel.pth"))
lstm_model.eval()

class SimpleInflationPredictor(torch.nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(SimpleInflationPredictor, self).__init__()
        layers = [torch.nn.Linear(input_size, num_neurons), torch.nn.ReLU()]
        for _ in range(1, num_layers):
            layers += [torch.nn.Linear(num_neurons, num_neurons), torch.nn.ReLU()]
        layers += [torch.nn.Linear(num_neurons, 1)]
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

nn_3inputs_scaler = joblib.load("./Backend/SavedModels/3inmodelscaler.gz")
nn_3inputs_model = SimpleInflationPredictor(input_size=3, num_layers=2, num_neurons=56)
nn_3inputs_model.load_state_dict(torch.load("./Backend/SavedModels/3inmodel.pth"))
nn_3inputs_model.eval()

class InflationPredictor(torch.nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(InflationPredictor, self).__init__()
        layers = [torch.nn.Linear(input_size, num_neurons), torch.nn.ReLU()]
        for _ in range(1, num_layers):
            layers += [torch.nn.Linear(num_neurons, num_neurons), torch.nn.ReLU()]
        layers += [torch.nn.Linear(num_neurons, 1)]
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

nn_9inputs_scaler = joblib.load("./Backend/SavedModels/9inmodelscaler.gz")
nn_9inputs_model = InflationPredictor(input_size=9, num_layers=3, num_neurons=98)
nn_9inputs_model.load_state_dict(torch.load("./Backend/SavedModels/9inmodel.pth"))
nn_9inputs_model.eval()


def predict_arima(features, months):
    arima_model = joblib.load('./Backend/SavedModels/arima_model_212.pkl')
    prediction = arima_model.forecast(steps=months)
    formatted_prediction = f"{prediction.iloc[-1]:.2f}%"  
    return formatted_prediction

def predict_lstm(features):
    features_scaled = lstm_scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    prediction = lstm_model(features_tensor).item()
    formatted_prediction = f"{prediction:.2f}%"
    return formatted_prediction

def predict_nn_3(features):
    features_scaled = nn_3inputs_scaler.transform(features)
    prediction = nn_3inputs_model(torch.tensor(features_scaled, dtype=torch.float32)).item()
    formatted_prediction = f"{prediction:.2f}%"
    return formatted_prediction

def predict_nn_9(features):
    features_scaled = nn_9inputs_scaler.transform(features)
    prediction = nn_9inputs_model(torch.tensor(features_scaled, dtype=torch.float32)).item()
    formatted_prediction = f"{prediction:.2f}%"
    return formatted_prediction

def predict_rnn(features):
    features_scaled = rnn_scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = rnn_model(features_tensor).item()
    formatted_prediction = f"{prediction:.2f}%"
    return formatted_prediction

@app.post("/predict/")
async def predict_inflation(data: schemas.PredictionRequest):
    model_name = data.model_name
    features = np.array(data.features).reshape(1, -1)
    if model_name == "LSTM":
        prediction = predict_lstm(features)
    elif model_name == "NN_3":
        prediction = predict_nn_3(features)
    elif model_name == "NN_9":
        prediction = predict_nn_9(features)
    elif model_name =="RNN":
        prediction = predict_rnn(features)
    else:
        raise HTTPException(status_code=400, detail="Model not supported")

    return {"predicted_inflation": prediction}

@app.post("/predict/arima/")
async def predict_arima(request: ARIMAPredictionRequest):
    arima_model = joblib.load('./Backend/SavedModels/arima_model_212.pkl')
    predictions = arima_model.forecast(steps=request.months)
    formatted_predictions = [f"{pred:.2f}%" for pred in predictions]
    return {"predicted_inflation": formatted_predictions}

@app.post("/compare_models/")
async def compare_models(data: ComparisonRequest):
    features = np.array(data.features).reshape(1, -1)
    
    nn3_features = features[:, :3]  # NN (3 inputs) uses only the first 3 features
    nn9_features = features
    lstm_features = features

    nn3_prediction = predict_nn_3(nn3_features)
    nn9_prediction = predict_nn_9(nn9_features)
    lstm_prediction = predict_lstm(lstm_features)

    return {
        "nn3_prediction": nn3_prediction,
        "nn9_prediction": nn9_prediction,
        "lstm_prediction": lstm_prediction
    }

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.error("Login failed: Incorrect username or password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"Access token created for user: {user.username}")
    return {"access_token": access_token, "token_type": "Bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).filter(User.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user

logger = logging.getLogger(__name__)

@app.post("/send_data")
async def send_data(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        user = await get_current_user(token, db)
    except Exception as e:
        logger.error(f"Error in user authentication: {e}", exc_info=True)
        return JSONResponse(status_code=401, content={"detail": "User not authenticated"})
    
    csv_file_path = './Backend/Data/complete_data.csv'
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found at path: {csv_file_path}", exc_info=True)
        return JSONResponse(status_code=404, content={"detail": "CSV file not found"})

    sender_email = "gigel.parcurel@yahoo.com"
    receiver_email = user.email
    subject = "Macroeconomic Data"
    body = "Please find the attached macroeconomic data CSV file."

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    try:
        with open(csv_file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename=complete_data.csv")
            message.attach(part)
    except Exception as e:
        logger.error(f"Error attaching file: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to attach file"})

    try:
        email_password = "lxwbhcalxdxwpraf"
        await aiosmtplib.send(
            message,
            hostname="smtp.mail.yahoo.com",
            port=587,
            start_tls=True,
            username=sender_email,
            password=email_password,
        )
        logger.info("Email sent successfully")
        return JSONResponse(status_code=200, content={"message": "Email sent successfully"})
    except aiosmtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to send email"})
    except Exception as e:
        logger.error(f"Error sending email: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to send email"})
    
@app.post("/send_model_scaler")
async def send_model_scaler(request: Request, data: schemas.ModelRequest, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        user = await get_current_user(token, db)
    except Exception as e:
        logger.error(f"Error in user authentication: {e}", exc_info=True)
        return JSONResponse(status_code=401, content={"detail": "User not authenticated"})
    
    model_name = data.model_name
    model_path = ""
    scaler_path = ""
    if model_name == "ARIMA":
        model_path = "./Backend/SavedModels/arima_model_212.pkl"
        scaler_path = "./Backend/SavedModels/rnn_scaler.gz"
    elif model_name == "LSTM":
        model_path = "./Backend/SavedModels/lstmmodel.pth"
        scaler_path = "./Backend/SavedModels/lstmscaler.gz"
    elif model_name == "NN_3":
        model_path = "./Backend/SavedModels/3inmodel.pth"
        scaler_path = "./Backend/SavedModels/3inmodelscaler.gz"
    elif model_name == "NN_9":
        model_path = "./Backend/SavedModels/9inmodel.pth"
        scaler_path = "./Backend/SavedModels/9inmodelscaler.gz"
    elif model_name == "RNN":
        model_path = "./Backend/SavedModels/rnn_model.pth"
        scaler_path = "./Backend/SavedModels/rnn_scaler.gz"
    else:
        return JSONResponse(status_code=400, content={"detail": "Model not supported"})

    sender_email = "gigel.parcurel@yahoo.com"
    receiver_email = user.email
    subject = f"{model_name} Model and Scaler"
    body = f"Please find the attached {model_name} model and scaler files."

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    try:
        with open(model_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={model_name}_model.pth")
            message.attach(part)
    except Exception as e:
        logger.error(f"Error attaching model file: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to attach model file"})

    try:
        with open(scaler_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={model_name}_scaler.gz")
            message.attach(part)
    except Exception as e:
        logger.error(f"Error attaching scaler file: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to attach scaler file"})

    try:
        email_password = "lxwbhcalxdxwpraf"
        await aiosmtplib.send(
            message,
            hostname="smtp.mail.yahoo.com",
            port=587,
            start_tls=True,
            username=sender_email,
            password=email_password,
        )
        logger.info("Email sent successfully")
        return JSONResponse(status_code=200, content={"message": "Email sent successfully"})
    except aiosmtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to send email"})
    except Exception as e:
        logger.error(f"Error sending email: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Failed to send email"})
    

@app.get("/current_user", response_class=JSONResponse)
async def get_current_user_info(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        user = await get_current_user(token, db)
        logging.info(f"Current user: {user.username}")
        return {"username": user.username}
    except HTTPException as e:
        logging.error(f"HTTPException: {e}")
        return {"username": "Guest"}
    except Exception as e:
        logging.error(f"Exception: {e}")
        return {"username": "Guest"}
