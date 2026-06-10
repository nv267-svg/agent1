import pandas as pd 
import sqlite3 
import requests 

df=pd.read_csv("crop_yield.csv") 

dat=sqlite3.connect("crop.db")

df.to_sql(
    "crop",dat,if_exists="replace",index=False
)

