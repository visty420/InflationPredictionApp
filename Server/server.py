from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

app = FastAPI()
app.mount("/static", StaticFiles(directory="Frontend"), name="static")
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
    
@app.get("/view_data")
async def view_data():
    import pandas as pd
    df = pd.read_csv("./Backend/Data/complete_data.csv")
    return HTMLResponse(content=df.to_html(classes="data", border=0))


arima_model = joblib.load("./Backend/SavedModels/arimamodel.pkl")
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
    arima_model = joblib.load('./Backend/SavedModels/arimamodel.pkl')
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
    arima_model = joblib.load('./Backend/SavedModels/arimamodel.pkl')
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