import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
 
 
 
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


import spacy
nlp = spacy.load('en')
processed_docs = []    
for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    print(len(doc))

    # Remove common words from a stopword list.
    #doc = [token for token in doc if token not in STOPWORDS]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)
    
docs = processed_docs
df_pdocs=pd.DataFrame(docs)
df_pdocs.to_csv('pdoc.csv')