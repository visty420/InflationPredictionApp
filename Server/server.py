from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import crud, models, schemas, auth

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
async def register_user(request: Request, username: str = Form(...), password: str = Form(...), email: str = Form(...)):
    user_data = schemas.UserCreate(username=username, email=email, password=password)
    new_user = await crud.create_user(user_data)
    if new_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    raise HTTPException(status_code=400, detail="Error creating user")