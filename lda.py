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


def load_data():
    print('\nLoading corpus, dictionary: ')
    dictionary=pickle.load(open("dic.p","rb"))
    corpus=pickle.load(open("corpus.p","rb"))
    print("Loading Completed")
    return dictionary, corpus

def load_dictionary(fname="dic.p"):
    print('\nLoading dictionary: ')
    dictionary=pickle.load(open(fname,"rb"))
    print("Loading Completed")
    return dictionary

def load_corpus(fname="corpus.p"):
    print('\nLoading Corpus: ')
    corpus=pickle.load(open(fname,"rb"))
    print("Loading Completed")
    return corpus

def load_doc_vecs(fname="doc_vecs.p"):
    print('\nLoading docs_vecs: ')
    doc_vecs=pickle.load(open(fname,"rb"))
    print("Loading Completed")
    return doc_vecs

def load_LDA_model(name='nips.lda'):
    
    nips=gensim.models.LdaModel.load(name)
    return nips
    
    


def get_vector_similarity_hellinger(vec1, vec2,model):
    '''Get similarity between two vectors'''
    
    dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics), \
                              matutils.sparse2full(vec2, model.num_topics))
    sim = 1.0 / (1.0 + dist)
    return sim

def get_sims(vec,vecs,model='load'):
    ''''Get similarity between a vector and a collection of vectors'''
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    
    sims = [get_vector_similarity_hellinger(vec, vec2,model) for vec2 in vecs]
    return sims

def get_doc_sim_table_from_doc_num(doc_num, top_n=10,model='load',corpus='load',doc_vecs='load'):
    
    if(corpus=='load'):
        corpus=load_corpus()
        
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    
    if(doc_vecs=='load'):
        print('Loading saved doc_vecs from doc_vecs.p')
        doc_vecs=load_doc_vecs()
    
    sims = get_sims(model.get_document_topics(corpus[doc_num]),doc_vecs,model)
    table = []
    for elem in enumerate(sims):
        sim = elem[1]
        table.append((elem[0], sim))
    df = pd.DataFrame(table, columns=['DocId', 'Score'])
    df = df.sort_values('Score', ascending=False)[:top_n]
    
    return df


def get_doc_topic_distribution(doc_num,model,doc_vecs='load'):
    if(doc_vecs=='load'):
        doc_vecs=load_doc_vecs()
    return matutils.sparse2full(doc_vecs[doc_num], model.num_topics)
    

def create_LDA_model(corpus,dictionary,num_topics,alpha='auto'):
    print('\nSetting number of topics to :%d ' %num_topics)
    print('\nCreating Model')
    model = gensim.models.LdaModel(corpus, id2word=dictionary, alpha=alpha, num_topics=num_topics)
    print('\nSavind Model as nips.lda')
    model.save('nips.lda')
    print('\nmodel saved')
    
    
def create_doc_vecs(model='load',corpus='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    if(corpus=='load'):
        corpus=load_corpus()
    
    
    doc_vecs = [model.get_document_topics(bow) for bow in corpus]
    pickle.dump(doc_vecs,open("doc_vecs.p","wb"))
    print('\nSaving doc_vecs')




    
def print_top_titles_by_topic(model='load',topic_labels='not_assigned',doc_vecs='load',df='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i+1) for i in range(model.num_topics)]
    if(doc_vecs=='load'):
        print('Loading saved doc_vecs from doc_vecs.p')
        doc_vecs=load_doc_vecs()
    if(df=='load'):
        print('Loading df_papers from database')
        database = "sqlite:///database.sqlite"
        engine=create_engine(database)
        conn = engine.connect()
        df=pd.read_sql_table('papers',conn) 
        
    
    
    sparse_vecs=np.array([matutils.sparse2full(vec, model.num_topics) for vec in doc_vecs])
    top_idx = np.argsort(sparse_vecs,axis=0)[-3:]
    count = 0
    for idxs in top_idx.T: 
        print("\nTopic {}:".format(count))
        for idx in idxs:
            print(df.iloc[idx]['title'])
        count += 1
    

    
    
def ldaquery(query,dictionary='load',model='load'):
    if(dictionary=='load'):
        print('Loading saved dictionary from dic.p')
        dictionary=pickle.load(open("dic.p","rb"))
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
        
    id2word = gensim.corpora.Dictionary()
    _ = id2word.merge_with(dictionary)
    query = query.split()
    query = id2word.doc2bow(query)
    #print(model[query])
    a = list(sorted(model[query], key=lambda x: x[1]))
    x=model.print_topic(a[-1][0])
    print(x)
    return a[-1][0],model[query]

def plot_evolution_plots(doc_vecs='load',model='load',topic_labels='not_assigned',df='load',tsne_embedding='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    if(doc_vecs=='load'):
        print('Loading saved doc_vecs from doc_vecs.p')
        doc_vecs=load_doc_vecs()
    if(tsne_embedding=='load'):
        print('Loading saved tsne_embedding from tsne_embedding.p')
        tsne_embedding=pickle.load(open("tsne_embedding.p","rb"))
   
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i) for i in range(model.num_topics)]
    if(df=='load'):
        print('Loading df_papers from database')
        database = "sqlite:///database.sqlite"
        engine=create_engine(database)
        conn = engine.connect()
        df=pd.read_sql_table('papers',conn) 
    sparse_vecs=np.array([matutils.sparse2full(vec, model.num_topics) for vec in doc_vecs])
    #tsne = TSNE(random_state=3211)
    #tsne_embedding = tsne.fit_transform(sparse_vecs)
    tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
    tsne_embedding['hue'] = sparse_vecs.argmax(axis=1)
    colors=np.random.rand(model.num_topics,4)
    legend_list = []

    for i in range(len(topic_labels)):   
        color = colors[i]
        legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
    matplotlib.rc('font',family='monospace')
    plt.style.use('ggplot')
    
    
    fig, axs = plt.subplots(3,2, figsize=(10, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .1, wspace=0)
    
    axs = axs.ravel()
    for year, idx in zip([1991,1996,2001,2006,2011,2016], range(6)):
        data = tsne_embedding[df['year']<=year]
        _ = axs[idx].scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")
        axs[idx].set_title('published until {}'.format(year),**{'fontsize':'10'})
        axs[idx].axis('off')
    
    plt.suptitle("all NIPS proceedings clustered by topic",**{'fontsize':'14','weight':'bold'})
    plt.figtext(.51,0.95,'unsupervised topic modeling with NMF based on textual content + 2D-embedding with t-SNE:', **{'fontsize':'10','weight':'light'}, ha='center')
    
    
    fig.legend(legend_list,topic_labels,loc=(0.1,0.89),ncol=3)
    plt.subplots_adjust(top=0.85)
    
    plt.show()


def print_top_topics_of_year(year,doc_vecs='load',model='load',topic_labels='not_assigned',df='load',tsne_embedding='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    if(doc_vecs=='load'):
        print('Loading saved doc_vecs from doc_vecs.p')
        doc_vecs=load_doc_vecs()
    if(tsne_embedding=='load'):
        print('Loading saved tsne_embedding from tsne_embedding.p')
        tsne_embedding=pickle.load(open("tsne_embedding.p","rb"))
   
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i+1) for i in range(model.num_topics)]
    if(df=='load'):
        print('Loading df_papers from database')
        database = "sqlite:///database.sqlite"
        engine=create_engine(database)
        conn = engine.connect()
        df=pd.read_sql_table('papers',conn) 
    sparse_vecs=np.array([matutils.sparse2full(vec, model.num_topics) for vec in doc_vecs])
    #tsne = TSNE(random_state=3211)
    #tsne_embedding = tsne.fit_transform(sparse_vecs)
    #years=df['year'].values
    #df_dist=pd.DataFrame({'year':years,'Topic_Distribution':sparse_vecs})
    df_sp=pd.DataFrame(sparse_vecs)
    df_dist=df_sp[df['year']==year]
    top_topic=df_dist.sum().idxmax()
    value=df_dist.sum()[top_topic]
    print('Top topic: %s'% topic_labels[top_topic])
    print('Value: %f'%value)
    #data = df[df['year']<=year]
    ax=df_dist.sum().plot(kind='bar')
    ax.set_xticklabels(topic_labels, rotation=90)
    plt.title('Topic distribution for the year:%d' %year)
    plt.show()
    return df_dist
    
    
    
def create_topic_dist_for_all_years(doc_vecs='load',model='load',topic_labels='not_assigned',df='load',tsne_embedding='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    if(doc_vecs=='load'):
        print('Loading saved doc_vecs from doc_vecs.p')
        doc_vecs=load_doc_vecs()
    if(tsne_embedding=='load'):
        print('Loading saved tsne_embedding from tsne_embedding.p')
        tsne_embedding=pickle.load(open("tsne_embedding.p","rb"))
   
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i+1) for i in range(model.num_topics)]
    if(df=='load'):
        print('Loading df_papers from database')
        database = "sqlite:///database.sqlite"
        engine=create_engine(database)
        conn = engine.connect()
        df=pd.read_sql_table('papers',conn) 
    sparse_vecs=np.array([matutils.sparse2full(vec, model.num_topics) for vec in doc_vecs])
    years=[i for i in range(df['year'].min(),df['year'].max()+1)]
    df_sp=pd.DataFrame(sparse_vecs)    
    year_dist=[]
    for year in years:
        df_dist=df_sp[df['year']==year]
        top_topic=df_dist.sum().idxmax()
        value=df_dist.sum()[top_topic]
        print('Top topic: %s'% topic_labels[top_topic])
        print('Value: %f'%value)
        #data = df[df['year']<=year]
        ax=df_dist.sum().plot(kind='bar')
        ax.set_xticklabels(topic_labels, rotation=90)
        plt.title('Topic distribution for the year:%d' %year)
        plt.show()
        year_dist.append(df_dist)
    pickle.dump(year_dist,open("year_dist.p","wb"))
    return year_dist

def topic_evolution_by_year(topic):
    year_dist=pickle.load(open("year_dist.p","rb"))
    topic_score=[]
    for i in range(len(year_dist)):
        topic_score.append(year_dist[i].sum()[topic])
    tf=pd.DataFrame({'Topic'+str(topic):topic_score})
    ax=tf.plot(kind='bar')
    ax.set_xticklabels([y for y in range(1987,2017)], rotation=90)
        
    return topic_score


def plot_all_topic_evolutions(model='load'):
    if(model=='load'):
        print('Loading saved model from nips.lda')
        model=load_LDA_model()
    for i in range(model.num_topics):
        topic_evolution_by_year(i)
        

def get_author_docs(author_id,df_map):
    return df_map['paper_id'][df_map['author_id']==author_id].values


def get_author_topic_dist_for_year(a_id,year,model,df_map,df_papers,verbose=True,doc_vecs='load'):
    #print(year)
    if(doc_vecs=='load'):
        doc_vecs=load_doc_vecs()
    
    docs=get_author_docs(a_id,df_map)
    df=pd.concat([df_papers[df_papers['id']==d] for d in docs])
    df=df[df['year']==year]
    if(df.empty):
        if(verbose):
            print('Author didn\'t publish any papers in the year:%d' %year)
        return np.array([0]*model.num_topics)
    else:
      return np.sum([get_doc_topic_distribution(df_papers.index[df_papers['id']==df['id'].values[i]].tolist()[0],model,doc_vecs=doc_vecs) for i in range(len(df))],axis=0)
    #return df,df.empty
        
def plot_author_evolution_plot(a_id,model,df_map,df_papers,topic_labels='not_assigned',plot=True,doc_vecs='load'):
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i+1) for i in range(model.num_topics)]
    if(doc_vecs=='load'):
        doc_vecs=load_doc_vecs()
    
    years=[i for i in range(df_papers['year'].min(),df_papers['year'].max()+1)]
    df=pd.concat([pd.DataFrame({year:get_author_topic_dist_for_year(a_id,year,model,df_map,df_papers,verbose=plot,doc_vecs=doc_vecs)}) for year in years],axis=1)
    if(plot):
        
        for y in years:
            if(df[y].sum()==0):
                print('Author didn\'t publish any papers in the year:%d' %y)
            else:
                ax=df[y].plot(kind='bar')
                ax.set_xticklabels(topic_labels, rotation=90)
                plt.title('Topic Distribution of Author in the year: %d'%y)
                plt.show()
    return df
    
    
def plot_top_topic_of_author_by_year(df_author_eval,plot=True):
    years=[y for y in range(1987,2017)]
    dc=df_author_eval.cumsum(axis=1)
    p=[dc[y].idxmax() for y in years]
    if(plot):
        plt.bar(years,p)
        plt.show()
        print("The author changed the topic %d times" %len(np.unique(p)))
        print("\n Following are the topics:")
        for i in np.unique(p):
            print("\nTopic %d" %i)
        
    return p

def compute_top_topic_by_year_matrix_of_all_authors(lda_model,df_authors,df_map,df_papers,doc_vecs='load'):
    p=[]
    if(doc_vecs=='load'):
        doc_vecs=load_doc_vecs()
    
    for i in df_authors['id'].values:
        print(i)
        p.append(plot_top_topic_of_author_by_year(plot_author_evolution_plot(i,lda_model,df_map,df_papers,plot=False,doc_vecs=doc_vecs),plot=False))
    pickle.dump(p,open("Atop_topic_by_year.p","wb"))
    return p
        
        

def find_most_frequent_author(top_n=10):
    a=np.array(pickle.load(open("Atop_topic_by_year.p","rb")))
    f=[len(np.unique(a[i])) for i in range(len(a))]
    top_idx=np.flip(np.argsort(f)[-top_n:],axis=0)
    top_f=[f[i] for i in range(len(top_idx))]
    top_name=[df_authors['name'][i] for i in range(len(top_idx))]
    df=pd.DataFrame({'Author':top_name,'Frequency':top_f})
    return df
    
    
    
    
    
if __name__=='__main__':
    num_topics=8
    #print(get_doc_sim_table_from_doc_num(12))
    #print_top_titles_by_topic()
    #plot_evolution_plots()
    #create_doc_vecs()
#    print('Creating LDA model: ')
#    dictionary,corpus=load_data()
#    create_LDA_model(corpus,dictionary,num_topics,alpha='auto')
#    nips=load_LDA_model()
#    i,q=ldaquery('kernel strategy')
    #td=get_doc_topic_distribution(12,nips)
    #df_dist=print_top_topics_of_year(1993)
    #year_dist=create_topic_dist_for_all_years()
    #ts=topic_evolution_by_year(7)
    #plot_all_topic_evolutions()
    #print(get_author_docs('13',df_map))
    #td=get_author_topic_dist_for_year('178',2016,nips,df_map,df_papers)
    #dtest=plot_author_evolution_plot('178',nips,df_map,df_papers)
    #p=plot_top_topic_of_author_by_year(dtest)
    Atop_topic_by_year=compute_top_topic_by_year_matrix_of_all_authors(nips,df_authors,df_map,df_papers)
    
    
    
    
    
    
