

# from fastapi import FastAPI
# from pydantic import BaseModel
# from prediction import predict_sentiment
# import uvicorn

# app = FastAPI()


# class TextData(BaseModel):
#     text: str

# @app.post("/predict/")
# async def predict(text_data: TextData):
#     sentiment = predict_sentiment(text_data.text)
#     return {"sentiment": sentiment}


from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prediction import predict_sentiment


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


class TextData(BaseModel):
    text: str

@app.post("/predict/", response_model=dict)
async def predict(text_data: TextData):
    sentiment = predict_sentiment(text_data.text)
    return {"sentiment": sentiment}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
