from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


app = FastAPI()


def extract_date(d):
    day = d.day_of_week
    month = d.month
    year = d.year
    return day,month,year

def extract_item_data(item):
    x = item.split('_')
    dept_id = '_'.join([x[0],x[1]])
    cat_id = x[0]

    return dept_id, cat_id

def extract_state(store):
    x = store.split('_')
    return x[0]

def merge_events(df):
    df_events = pd.read_csv('sources/clean_events.csv')
    df = pd.merge(df, df_events, left_on = ['date'], right_on = ['date'], how = 'left')
    df = df.fillna(0)
    del(df_events)
    return df


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'AMLA AT2 working.'

@app.get("/sales/stores/items")
def predict(item:str, store:str, date:str):

    d = pd.to_datetime(date, errors='coerce')
    if pd.isna(d):
        return 'invalid date detected. Please use YYYY-MM-DD format'
    


    return {"Hello": "World"}

@app.get("/sales/national")
def predict(date:str):

    d = pd.to_datetime(date, errors='coerce')
    if pd.isna(d):
        return 'invalid date detected. Please use YYYY-MM-DD format'


    return {"Hello": "World"}