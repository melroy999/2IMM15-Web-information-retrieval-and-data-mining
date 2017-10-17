import pickle
import gensim
from gensim import matutils
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import random 
random.seed(13)

#visualization packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
import lda

#stopwords.words('english')



def plot_number_of_papers_by_year(df):
    sns.countplot(x='year',data=df)
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=90)
    plt.show()
    

def plot_word_cloud_by_paper_id(paper_id,df_papers):
    wc=WordCloud(stopwords=stopwords.words('english')).generate(df_papers['paper_text'][paper_id].lower()) 
    plt.imshow(wc)
    plt.axis('off')


#plot_number_of_papers_by_year(df_papers)
#plot_word_cloud_by_paper_id(12,df_papers)

load=True
if(load):
    nips=lda.load_LDA_model()
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

    

        
    