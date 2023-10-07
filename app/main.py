from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


app = FastAPI()
gb = load('../models/predictive/xgb.joblib')

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

def merge_events(df, date):
    df_events = pd.read_csv('sources/clean_events.csv')
    event = df_events.loc[df_events['date'] == date]
    del(df_events)
    event = event[event.columns[1:]]
    df = df.join(event).fillna(0)
    df[event.columns] = df[event.columns].astype(int)
    
    #df = pd.merge(df, df_events, left_on = ['date'], right_on = ['date'], how = 'left')
    #df = df.fillna(0)

    df.drop('date')
    
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
    
    dept_id, cat_id = extract_item_data(item)
    day, month, year = extract_date(d)

    input = [{
        "item_id": item,
        "dept_id": dept_id,
        "cat_id": cat_id,
        "store_id": store,
        "state_id": extract_state(store),
        "day_of_week": day,
        "month": month,
        "year": year,
        "date": d

    }]
    df = pd.DataFrame.from_dict(input)
    df = merge_events(df, date)
    
    oe = load('../models/item_encoder.joblib')
    ohe = load('../models/ohe_encorder.joblib')

    ## perform encoding
    df[['item_id','dept_id', 'store_id']] = oe.transform(df[['item_id','dept_id', 'store_id']])
    test = ohe.transform(df[['cat_id', 'state_id']])

    ##merge encoded columns
    num_cols = ['event_cultural', 'event_national', 'event_religious', 'event_sport', 'day_of_week', 'month', 'year']
    test = test.join(df[['item_id','dept_id', 'store_id']])
    test = test.join(df[num_cols])
    del(oe)
    del(ohe)

    pred = gb.predict(test)
    pred = round(pred[0],5)
    
    return {"item": item, 'store': store, 'date': date, 'pred': pred}

@app.get("/sales/national")
def predict(date:str):

    d = pd.to_datetime(date, errors='coerce')
    if pd.isna(d):
        return 'invalid date detected. Please use YYYY-MM-DD format'


    return {"Hello": "World"}