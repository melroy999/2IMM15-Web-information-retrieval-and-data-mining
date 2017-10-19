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

class LDA_TM:
    
    def __init__(self,model_name):
        database = "sqlite:///database.sqlite"
        engine=create_engine(database)
        conn = engine.connect()
        #with conn:
        self.df_papers=pd.read_sql_table('papers',conn) 
        self.df_authors=pd.read_sql_table('authors',conn) 
        self.df_map=pd.read_sql_table('paper_authors',conn) 
        self.df_papers.id=self.df_papers.id.astype(str)
        self.df_map.id=self.df_map.id.astype(str)
        self.df_map.paper_id=self.df_map.paper_id.astype(str)
        self.df_map.author_id=self.df_map.author_id.astype(str)
        self.df_authors.id=self.df_authors.id.astype(str)
        self.model_name=model_name+'.lda'
        self.doc_vecs_name=model_name+'_doc_vecs.p'
        self.tsne_name=model_name+'_tsne.p'
        self.author_top_topic_by_year_name=model_name+'_author_top_topic_by_year.p'
        self.year_dist_name=model_name+'_year_dist.p'
        
        
    def load_existing_model(self):
        self.load_data()
        self.load_LDA_model(self.model_name)
        self.load_doc_vecs(self.doc_vecs_name)
        self.topic_labels=['Topic #'+str(i+1) for i in range(self.model.num_topics)]
        self.load_Tsne_embedding()
        self.year_dist=self.load_year_dist()
        self.load_author_top_topic_by_year_matrix()
        
        
        
        
    def load_data(self):
        print('\nLoading corpus, dictionary: ')
        self.dictionary=pickle.load(open("dic.p","rb"))
        self.corpus=pickle.load(open("corpus.p","rb"))
        print("Loading Completed")
        
    def load_dictionary(self,fname="dic.p"):
        print('\nLoading dictionary: ')
        self.dictionary=pickle.load(open(fname,"rb"))
        print("Loading Completed")
        
    def load_corpus(self,fname="corpus.p"):
        print('\nLoading Corpus: ')
        self.corpus=pickle.load(open(fname,"rb"))
        print("Loading Completed")
       
    def load_doc_vecs(self,fname="doc_vecs.p"):
        print('\nLoading docs_vecs: ')
        self.doc_vecs=pickle.load(open(fname,"rb"))
        print("Loading Completed")
        
    def load_LDA_model(self,name='nips.lda'):
        print("\nLoading model from %s" %self.model_name)
        
        self.model=gensim.models.LdaModel.load(name)
    
    def load_Tsne_embedding(self,name='tnse_embedding.p'):
        print("\nLoading Tsne embeddings from %s" %self.tsne_name)
        self.tsne_embedding=pickle.load(open(self.tsne_name,"rb"))
    
    def load_year_dist(self):
        print("\nLoading Year Distribution Matrix from %s" %self.year_dist_name)
        return pickle.load(open(self.year_dist_name,"rb"))
    
    def load_author_top_topic_by_year_matrix(self):
        print("\nLoading Author Top Topic of year Matrix from %s" %self.author_top_topic_by_year_name)
        self.author_top_topic_by_year=np.array(pickle.load(open(self.author_top_topic_by_year_name,"rb")))
        
    
        
    
    def create_LDA_model(self,num_topics,alpha='auto'):
        print('\nSetting number of topics to :%d ' %num_topics)
        print('\nCreating Model')
        self.model = gensim.models.LdaModel(self.corpus, id2word=self.dictionary, alpha=alpha, num_topics=num_topics)
        print('\nSavind Model as %s' %self.model_name)
        self.model.save(self.model_name)
        print('\nmodel saved')
        print('\nCreating Doc_vecs')
        self.create_doc_vecs()
        self.topic_labels=['Topic #'+str(i+1) for i in range(self.model.num_topics)]
        print('\nCreating Tsne Embeddings')
        self.create_tsne_embedding()
        print('\nCreating Year_dist')
        self.create_topic_dist_for_all_years()
        print('\nCreating Author Top Topic by Year Matrix')
        self.create_top_topic_by_year_matrix_of_all_authors()
        
        
        
    def create_doc_vecs(self):
        self.doc_vecs = [self.model.get_document_topics(bow) for bow in self.corpus]
        pickle.dump(self.doc_vecs,open(self.doc_vecs_name,"wb"))
        print('\nSaving doc_vecs')
    
    def create_tsne_embedding(self):
        sparse_vecs=np.array([matutils.sparse2full(vec, self.model.num_topics) for vec in self.doc_vecs])
        tsne = TSNE(random_state=3211)
        self.tsne_embedding = tsne.fit_transform(sparse_vecs)
        pickle.dump(self.tsne_embedding,open(self.tsne_name,"wb"))
        
    def create_topic_dist_for_all_years(self):
        model=self.model
        doc_vecs=self.doc_vecs
        topic_labels=self.topic_labels
        df=self.df_papers
        
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
        pickle.dump(year_dist,open(self.year_dist_name,"wb"))
        self.year_dist=year_dist
        return year_dist
    
    def create_top_topic_by_year_matrix_of_all_authors(self):
        p=[]
        for i in self.df_authors['id'].values:
            print(i)
            p.append(self.plot_top_topic_of_author_by_year(self.plot_author_evolution_plot(i,plot=False),plot=False))
        pickle.dump(p,open(self.author_top_topic_by_year_name,"wb"))
        return p

    
   
    def get_vector_similarity_hellinger(vec1, vec2,model):
        '''Get similarity between two vectors'''
        
        dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics), \
                                  matutils.sparse2full(vec2, model.num_topics))
        sim = 1.0 / (1.0 + dist)
        return sim
    
    def get_sims(self,vec,vecs):
        sims = [self.get_vector_similarity_hellinger(vec, vec2,self.model) for vec2 in vecs]
        return sims
    
    def get_doc_sim_table_from_doc_num(self,doc_num, top_n=10):
        
        
        sims = self.get_sims(self.model.get_document_topics(self.corpus[doc_num]),self.doc_vecs,self.model)
        table = []
        for elem in enumerate(sims):
            sim = elem[1]
            table.append((elem[0], sim))
        df = pd.DataFrame(table, columns=['DocId', 'Score'])
        df = df.sort_values('Score', ascending=False)[:top_n]
        
        return df
    
    
    def get_doc_topic_distribution(self,doc_num):
        return matutils.sparse2full(self.doc_vecs[doc_num], self.model.num_topics)
    
        
    def print_top_titles_by_topic(self):
        sparse_vecs=np.array([matutils.sparse2full(vec, self.model.num_topics) for vec in self.doc_vecs])
        top_idx = np.argsort(sparse_vecs,axis=0)[-3:]
        count = 0
        df=self.df_papers
        for idxs in top_idx.T: 
            print("\nTopic {}:".format(count))
            for idx in idxs:
                print(df.iloc[idx]['title'])
            count += 1
            
    def set_topic_labels(self,topic_labels):
        self.topic_labels=topic_labels
        
        
    
        
        
    def ldaquery(self,query):
            
        id2word = gensim.corpora.Dictionary()
        _ = id2word.merge_with(self.dictionary)
        query = query.split()
        query = id2word.doc2bow(query)
        #print(model[query])
        a = list(sorted(self.model[query], key=lambda x: x[1]))
        x=self.model.print_topic(a[-1][0])
        print(x)
        return a[-1][0],self.model[query]
    
        
        
    
    
    def plot_evolution_plots(self):
        model=self.model
        doc_vecs=self.doc_vecs
        topic_labels=self.topic_labels
        tsne_embedding=self.tsne_embedding
        df=self.df_papers
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
    
    
    def print_top_topics_of_year(self,year):
        model=self.model
        doc_vecs=self.doc_vecs
        topic_labels=self.topic_labels
        df=self.df_papers
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
        
        
        
    
    def topic_evolution_by_year(self,topic):
        year_dist=self.year_dist
        topic_score=[]
        for i in range(len(year_dist)):
            topic_score.append(year_dist[i].sum()[topic])
        tf=pd.DataFrame({'Topic'+str(topic):topic_score})
        ax=tf.plot(kind='bar')
        ax.set_xticklabels([y for y in range(1987,2017)], rotation=90)
            
        return topic_score
    
    
    
    def plot_all_topic_evolutions(self):
        model=self.model
        for i in range(model.num_topics):
            self.topic_evolution_by_year(i)
            
    
    def get_author_docs(self,author_id):
        'Returns the doucument Ids published by the author'
        return self.df_map['paper_id'][self.df_map['author_id']==author_id].values
    
    
    def get_author_topic_dist_for_year(self,a_id,year,verbose=True):
        'Sum of topic distributions of documents published for the given year'
        model=self.model
        df_papers=self.df_papers
        docs=self.get_author_docs(a_id)
        df=pd.concat([df_papers[df_papers['id']==d] for d in docs])
        df=df[df['year']==year]
        if(df.empty):
            if(verbose):
                print('Author didn\'t publish any papers in the year:%d' %year)
            return np.array([0]*model.num_topics)
        else:
          return np.sum([self.get_doc_topic_distribution(df_papers.index[df_papers['id']==df['id'].values[i]].tolist()[0]) for i in range(len(df))],axis=0)
        #return df,df.empty
            
    def plot_author_evolution_plot(self,a_id,plot=True):
        'Plots and returns the topic distribution of the author by year'
        df_papers=self.df_papers
        years=[i for i in range(df_papers['year'].min(),df_papers['year'].max()+1)]
        df=pd.concat([pd.DataFrame({year:self.get_author_topic_dist_for_year(a_id,year,verbose=plot)}) for year in years],axis=1)
        if(plot):
            
            for y in years:
                if(df[y].sum()==0):
                    print('Author didn\'t publish any papers in the year:%d' %y)
                else:
                    ax=df[y].plot(kind='bar')
                    ax.set_xticklabels(self.topic_labels, rotation=90)
                    plt.title('Topic Distribution of Author in the year: %d'%y)
                    plt.show()
        return df
        
        
    def plot_top_topic_of_author_by_year(self,df_author_eval,plot=True):
        '''needs output from plot_author_evolution_plot(a_id) as input
        Plots the top topic of the author for the given year'''
        
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
    
    
    def find_most_frequent_authors(self,top_n=10):
        'Returns the dataFrame of authors who changed topics most times'
        a=self.author_top_topic_by_year
        f=[len(np.unique(a[i])) for i in range(len(a))]
        top_idx=np.flip(np.argsort(f)[-top_n:],axis=0)
        top_f=[f[top_idx[i]] for i in range(len(top_idx))]
        top_name=[self.df_authors['name'][top_idx[i]] for i in range(len(top_idx))]
        top_id=[self.df_authors['id'][top_idx[i]] for i in range(len(top_idx))]

        df=pd.DataFrame({'Id':top_id,'Author':top_name,'Frequency':top_f})
        return df
        
    

if __name__=='__main__':
    Tm=LDA_TM('nipsLDA')
    #Tm.create_LDA_model(10)
    Tm.load_existing_model()
    #Tm.plot_author_evolution_plot('13')
    #Tm.get_author_topic_dist_for_year('13',1993)
    #print(Tm.get_author_docs('13'))
    #Tm.plot_all_topic_evolutions()
    #Tm.print_top_topics_of_year(1993)
    #Tm.print_top_titles_by_topic()
    #df=Tm.find_most_frequent_authors()

    
    

        
    
        
    
    