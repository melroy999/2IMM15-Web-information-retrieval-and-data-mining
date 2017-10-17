import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

import sqlite3

from sqlalchemy import create_engine


#df=pd.read_csv("authors.csv")
#df=pd.read_csv("paper_authors.csv")

import sqlite3
from sqlite3 import Error
 
 
 
database = "sqlite:///database.sqlite"
 
# create a database connection
engine=create_engine(database)
conn = engine.connect()
#with conn:
df_papers=pd.read_sql_table('papers',conn) 
df_authors=pd.read_sql_table('authors',conn) 
df_map=pd.read_sql_table('paper_authors',conn) 


    
