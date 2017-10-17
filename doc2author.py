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
p_id=df_papers['id'].values
a_name=df_authors['name'].values

doc2an=[]

for i in range(len(p_id)):
    k=df_map.paper_id[df_map.paper_id==p_id[i]].index.tolist()
    doc2an.append(df_map['author_id'].values[k].tolist())

df_doc2a=pd.DataFrame({'paper_id':p_id,'paper_authors':doc2an})
doc2author=df_doc2a.set_index('paper_id').to_dict()['paper_authors']
author_ids=df_papers['id'].values

#doc_id_dict = dict(zip(author_ids, range(len(author_ids))))
## Replace NIPS IDs by integer IDs.
#for a, a_doc_ids in author2doc.items():
#    print(a)
#    print(a_doc_ids)
#    for i, doc_id in enumerate(a_doc_ids):
#        print(i)
#        print(doc_id)
#        author2doc[a][i] = doc_id_dict[doc_id]
#print(len(author2doc))

    