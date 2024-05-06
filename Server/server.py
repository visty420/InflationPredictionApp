from dbm import _Database
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from . import crud, models, schemas, auth
from database import get_db
from database import async_session

app = FastAPI()
app.mount("/static", StaticFiles(directory="Frontend"), name="static")
templates = Jinja2Templates(directory="Frontend/templates")


@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("mainpage.html", {"request": request})

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})



@app.post("/register")
async def register_user(
    request: Request, 
    username: str = Form(...), 
    password: str = Form(...), 
    email: str = Form(...),
    db: AsyncSession = Depends(get_db)  # Înlocuiește _Database.get_db cu get_db
):
    user_data = schemas.UserCreate(username=username, email=email, password=password)
    new_user = await crud.create_user(db, user_data)
    return {"username": new_user.username, "email": new_user.email}