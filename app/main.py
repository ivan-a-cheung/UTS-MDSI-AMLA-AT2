from fastapi import FastAPI
from joblib import load
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


app = FastAPI()
gb = load('../models/predictive/xgb.joblib')
fc = load('../models/forecasting/forecast.joblib')

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
    #df.drop('date')
    
    return df


@app.get("/", response_class='PlainTextResponse')
def read_root():
    return '''This app has been developed to predict revenue across the company stores. Two predictive calcualtions are available:
    1) predict expected revenue of an item in a given store, by date
    2) predict a 7 day forecast of total company revenue by nominated date

    The API endpoints are:
    (GET) / - this page
    (GET) /health/ - confirms status of the app
    (GET) /sales/national?date=YYYY-MM-DD returns a 7 day forecast of company revenue, starting from input date. if no date is provided, forecast will be applied on today's date
    (GET) /sales/stores/items/?item=____&store=____&date=YYYY-MM-DD     Predicts expected revenue from items sold at store on the date. All parameters are required for this endpoint to function.
    '''

@app.get('/health', status_code=200)
def healthcheck():
    return 'Welcome! AMLA AT2 is working.'

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
    ohe = load('../models/ohe_encoder.joblib')

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

    del(input)
    del(df)
    del(dept_id, cat_id)
    del(day, month, year)
    del(d)
    
    return {"item": item, 'store': store, 'date': date, 'predicted_revenue': pred}

@app.get("/sales/national")
def forecast(date:str = ''):
    if(date == ''):
        d = pd.to_datetime('today')
    else:
        d = pd.to_datetime(date, errors='coerce')
        if pd.isna(d):
            return 'invalid date detected. Please use YYYY-MM-DD format'

    dict = []
    for num in range(0,7):
        dict.append(d + pd.DateOffset(days=num))

    df = pd.DataFrame().from_dict(dict).rename(columns={0:"date"})
    del(d)
    del(dict)

    df['day_of_year'] = df['date'].dt.day_of_year
    df['day_of_week'] = df['date'].dt.day_of_week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    ##temp // test
    df['event_cultural'] = 0
    df['event_national'] = 0
    df['event_religious'] = 0
    df['event_sport'] = 0

    ## reorder
    results = pd.DataFrame(df.pop('date'))
    df = df[['event_cultural','event_national','event_religious','event_sport','day_of_year','day_of_week','month','year']]

    results['predicted revenue'] = fc.predict(df)
    
    results['date'] = results['date'].astype(str)
    return results.to_json(orient='records')