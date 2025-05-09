from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.predict import predict_labels

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, user_input: str = Form(...)):
    labels, probas = predict_labels(user_input)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "user_input": user_input,
        "result": labels,
        "probabilities": probas
    })
