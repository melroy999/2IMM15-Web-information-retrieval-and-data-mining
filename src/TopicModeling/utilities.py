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

def create_dictionary_and_corpus():
    print('\n Creating Dictionary')
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
    
    a2docn=[]
    
    for i in range(len(a_id)):
        k=df_map.author_id[df_map.author_id==a_id[i]].index.tolist()
        a2docn.append(df_map['paper_id'].values[k].tolist())
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
    from gensim.corpora import Dictionary
    import pickle
    dictionary = Dictionary(docs)
    print(len(dictionary))
    
    # Remove rare and common tokens.
    # Filter out words that occur too frequently or too rarely.
    max_freq = 0.5
    min_wordcount = 20
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    
    _ = dictionary[0]  # This sort of "initializes" dictionary.id2token.
    
    pickle.dump(dictionary,open("dic.p","wb"))
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    pickle.dump(corpus,open("corpus.p","wb"))
    

def create_author2docFile():
    print('\n Creating Author2Doc file')
    database = "sqlite:///database.sqlite"
    engine=create_engine(database)
    conn = engine.connect()
    df_papers=pd.read_sql_table('papers',conn) 
    df_authors=pd.read_sql_table('authors',conn) 
    df_map=pd.read_sql_table('paper_authors',conn) 
    
    df_papers.id=df_papers.id.astype(str)
    df_map.id=df_map.id.astype(str)
    df_map.paper_id=df_map.paper_id.astype(str)
    df_map.author_id=df_map.author_id.astype(str)
    df_authors.id=df_authors.id.astype(str)
    a_id=df_authors['id'].values
    a2docn=[]
    for i in range(len(a_id)):
        k=df_map.author_id[df_map.author_id==a_id[i]].index.tolist()
        a2docn.append(df_map['paper_id'].values[k].tolist())
    
    df_a2doc=pd.DataFrame({'author_id':a_id,'author_papers':a2docn})
    author2doc=df_a2doc.set_index('author_id').to_dict()['author_papers']
    doc_ids=df_papers['id'].values
    
    doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
    # Replace NIPS IDs by integer IDs.
    for a, a_doc_ids in author2doc.items():
        print(a)
        print(a_doc_ids)
        for i, doc_id in enumerate(a_doc_ids):
            print(i)
            print(doc_id)
            author2doc[a][i] = doc_id_dict[doc_id]
    print(len(author2doc))
    pickle.dump(author2doc,open("author2docModel.p","wb"))

    
    
    

    


#plot_number_of_papers_by_year(df_papers)
#plot_word_cloud_by_paper_id(12,df_papers)

load=False
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
	create_dictionary_and_corpus()
	create_author2docFile()

    

        
    