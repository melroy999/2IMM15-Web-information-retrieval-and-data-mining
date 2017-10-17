import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

from sqlalchemy import create_engine
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
docs=df_papers['paper_text'].values

df_papers.id=df_papers.id.astype(str)
df_map.id=df_map.id.astype(str)
df_map.paper_id=df_map.paper_id.astype(str)
df_map.author_id=df_map.author_id.astype(str)
df_authors.id=df_authors.id.astype(str)


a_id=df_authors['id'].values
a_name=df_authors['name'].values

a2docn=[]

for i in range(len(a_id)):
    k=df_map.author_id[df_map.author_id==a_id[i]].index.tolist()
    a2docn.append(df_map['paper_id'].values[k].tolist())

df_a2doc=pd.DataFrame({'author_id':a_id,'author_papers':a2docn})
df_aname2doc=pd.DataFrame({'author_id':a_name,'author_papers':a2docn})
author2doc=df_a2doc.set_index('author_id').to_dict()['author_papers']
doc_ids=df_papers['id'].values

#doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
## Replace NIPS IDs by integer IDs.
#for a, a_doc_ids in author2doc.items():
#    print(a)
#    print(a_doc_ids)
#    for i, doc_id in enumerate(a_doc_ids):
#        print(i)
#        print(doc_id)
#        author2doc[a][i] = doc_id_dict[doc_id]
#print(len(author2doc))

    