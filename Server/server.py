from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Setup pentru servirea fișierelor statice
app.mount("/static", StaticFiles(directory="Frontend"), name="static")

# Setup pentru templating
templates = Jinja2Templates(directory="Frontend/templates")

# Rute pentru paginile web
@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("mainpage.html", {"request": request})

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})
