from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'SGDClassifier is all ready to go!'

@app.get("/cvd/risks/prediction")
def predict():
    return {"Hello": "World"}
