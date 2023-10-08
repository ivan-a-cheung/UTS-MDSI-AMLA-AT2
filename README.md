36120 Advanced Machine Learning Application - Assignment 2
==============================

Heroku App: https://mysterious-sands-98640-63a0822c91e6.herokuapp.com/

<br/>

**Available Endpoints:**

A) (GET) /

B) (GET) /health

C) (GET) /national

D) (GET) /sales/stores/items

----------------------------------------------------

**Using Endpoints**

<br/>

_C) (GET) /national_

**Function:** This endpoint will return a JSON response with 7 days of national revenue forecast. This endpoint uses the [Forecast Model]

**Accepted Parameter:**

_Date_    This endpoint accepts a date variable under the format YYYY-MM-DD. If no date is provided, the current date will be used.

**Example request:** https://mysterious-sands-98640-63a0822c91e6.herokuapp.com/sales/national?date=2015-06-21

<br/>

_D) (GET) /sales/stores/items_

**Function:** This endpoint predicts the expected revenue of an item at a given store and date.

**Accepted Parameter:**

_Date_    This endpoint requires a date variable under the format YYYY-MM-DD.

_Store_   This endpoint requires the ID of the store

_Item_    This endpoint requires the ID of the item 

Note: all parameters are required for this endpoint to return a prediction.

**Example request:** https://mysterious-sands-98640-63a0822c91e6.herokuapp.com/sales/stores/items?date=2015-06-21&store=CA_1&item=HOBBIES_1_001

<br/>

----------------------------------------------------

