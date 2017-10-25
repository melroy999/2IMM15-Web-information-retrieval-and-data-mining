import pickle
import gensim
from gensim import matutils
from gensim.models import AuthorTopicModel

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#visualization packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.callbacks import EarlyStopping
from keras.models import load_model
import difflib

from bokeh.io import output_file
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.charts import Bar,show
        

class A_TM:
    
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
        self.load_data()
        self.load_author2doc()
        self.model_name=model_name+'.atmodel'
        self.author_vecs_name=model_name+'_author_vecs.p'
        self.tsne_name=model_name+'_tsne.p'
        self.kmeans_name=model_name+'_kmeans.p'
        self.keras_cl_model_name=model_name+'_keras.p'
        
    
        
        
    def load_existing_model(self):
        self.load_AT_model(self.model_name)
        self.load_author_vecs(self.author_vecs_name)
        self.topic_labels=['Topic #'+str(i+1) for i in range(self.model.num_topics)]
        self.load_tsne_embeddings()
        #self.year_dist=self.load_year_dist()
        #self.load_author_top_topic_by_year_matrix()
        self.load_kmeans()
        #self.load_classification_model()
        
        
        
        
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
    
    def load_author2doc(self,fname="author2docModel.p"):
        print('\nLoading author2doc')
        self.author2doc=pickle.load(open(fname,"rb"))
        print("Loading Completed")

       
    def load_author_vecs(self,fname="author_vecs.p"):
        print('\nLoading author_vecs: ')
        self.author_vecs=pickle.load(open(fname,"rb"))
        print("Loading Completed")
        
    def load_AT_model(self,name='model.atmodel'):
        print("\nLoading model from %s" %self.model_name)
        
        self.model=AuthorTopicModel.load(self.model_name)
    
    def load_kmeans(self):
        print("\nLoading Kmeans Clustering data from %s" %self.kmeans_name)
        self.kmeans= pickle.load(open(self.kmeans_name,"rb"))
        
    def load_classification_model(self):
        print("\nLoading Keras classification model from %s" %self.keras_cl_model_name)
        self.keras_cl_model= load_model(self.keras_cl_model_name)
    
    def load_tsne_embeddings(self):
        print('\n Loading tsne_embeddings')
        self.tsne=pickle.load(open(self.tsne_name,"rb"))
    
    
    def create_tsne_embeddings(self,smallest_author=1):
        model=self.model
        tsne = TSNE(n_components=2, random_state=0)
        authors = [model.author2id[a] for a in model.author2id.keys() if len(model.author2doc[a]) >= smallest_author]
        _ = tsne.fit_transform(model.state.gamma[authors, :])  # Result stored in tsne.embedding_
        self.tsne=tsne
        pickle.dump(tsne,open(self.tsne_name,"wb"))
    
    
        
        
        
    def create_author_vecs(self):
        self.author_vecs = [self.model.get_author_topics(author) for author in self.model.id2author.values()]
        pickle.dump(self.author_vecs,open(self.author_vecs_name,"wb"))
        print('\nSaving author_vecs')
    
    def create_AT_model(self,num_topics):
        corpus=self.corpus
        dictionary=self.dictionary
        author2doc=self.author2doc
        model_list = []
        for i in range(1):
            print(i)
            model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=dictionary.id2token, \
                            author2doc=author2doc, chunksize=2000, passes=100, gamma_threshold=1e-10, \
                            eval_every=0, iterations=1, random_state=i)
            top_topics = model.top_topics(corpus)
            tc = sum([t[1] for t in top_topics])
            model_list.append((model, tc))
        model, tc = max(model_list, key=lambda x: x[1])
        print('Topic coherence: %.3e' %tc)
        model.save(self.model_name)
        print('AT Model saved as %s' %self.model_name)
        self.model=model
        print('Creating author Vecs')
        self.create_author_vecs()
        print('\n Creating Clustering:')
        self.create_author_clustering(self.model.num_topics)
#        print('\nCreating Classification from  cluster Data')
#        self.create_classification_from_cluster_data()
        print('\nCreating TSNE embeddings')
        self.create_tsne_embeddings()
        
    
    
    
    def plot_author_clustering_interia(self,max_cluster=100,min_cluster=3):
        nips=self.model
        author_vecs = self.author_vecs
        X=[matutils.sparse2full(author, nips.num_topics) for author in author_vecs]
        inertianew=[]
        scaler = StandardScaler()
        scaler.fit(X)
        X_new=scaler.transform(X)
        
        for i in range(min_cluster,max_cluster):
            print('\nCreating K means clusters with cluters=%d'%i)
            kmeans = KMeans(n_clusters=i, random_state=0).fit(X_new)
            inertianew.append(kmeans.inertia_)
        
        plt.plot(list(range(min_cluster,max_cluster)),inertianew)
        plt.show()
#        output_file("AuthorClustering.html")
#        p = figure(plot_width=400, plot_height=400,title='Inertia plot for varying cluster size')
#        p.line(list(range(min_cluster,max_cluster)), inertianew)
#        p.xaxis.axis_label ='Number of Clusters'
#        p.yaxis.axis_label='Inertia'
#        show(p)
#        return inertianew
    
    def create_author_clustering(self,n_clusters):
        print('\nCreating K means clusters with cluters=%d'%n_clusters)
        nips=self.model
        author_vecs = self.author_vecs
        X=[matutils.sparse2full(author, nips.num_topics) for author in author_vecs]
        scaler = StandardScaler()
        scaler.fit(X)
        X_new=scaler.transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_new)
        pickle.dump(self.kmeans,open(self.kmeans_name,'wb'))
    
    def create_classification_from_cluster_data(self):
        print('\nCreating deep learning classification')
        nips=self.model
        author_vecs = self.author_vecs
        X=[matutils.sparse2full(author, nips.num_topics) for author in author_vecs]
        scaler = StandardScaler()
        scaler.fit(X)
        X_new=scaler.transform(X)
        cluster_data=self.kmeans.predict(X_new)
        target=to_categorical(cluster_data)
        model=Sequential()
        model.add(Dense(100, activation='relu', input_shape = (self.model.num_topics,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(target.shape[1], activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        early_stopping_monitor = EarlyStopping(patience=5) 

        model.fit(X_new, target,validation_split=0.3,epochs=50,callbacks=[early_stopping_monitor])
        #pickle.dump(model,open(self.keras_cl_model_name,"wb"))
        self.keras_cl_model=model
        model.save(self.keras_cl_model_name)
        
        
        
        
        


    
   
    def get_vector_similarity_hellinger(self,vec1, vec2,model):
        '''Get similarity between two vectors'''
        
        dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics), \
                                  matutils.sparse2full(vec2, model.num_topics))
        sim = 1.0 / (1.0 + dist)
        return sim
    
    def get_sims(self,vec,vecs):
        sims = [self.get_vector_similarity_hellinger(vec, vec2,self.model) for vec2 in vecs]
        return sims
    
    def get_sim_author_table(self,Id, top_n=10,smallest_author=1):
        author_vecs=self.author_vecs
        model=self.model
        sims = self.get_sims(model.get_author_topics(Id),author_vecs)
        table = []
        for elem in enumerate(sims):
            author_id = model.id2author[elem[0]]
            author_name=self.get_author_name_from_id(author_id)
            sim = elem[1]
            author_size = len(model.author2doc[author_id])
            if author_size >= smallest_author:
                table.append((author_id,author_name, sim, author_size))
                
        # Make dataframe and retrieve top authors.
        df = pd.DataFrame(table, columns=['Author_ID','Author_Name','Score', 'Size'])
        df = df.sort_values('Score', ascending=False)[:top_n]
        
        return df
    
    
    def printtopics(self):
        for topic in self.model.show_topics(num_topics=self.model.num_topics):
            print('Label: ' + self.topic_labels[topic[0]])
            words = ''
            for word, prob in self.model.show_topic(topic[0]):
                words += word + ' '
            print('Words: ' + words)
            
    def show_author_by_id(self,author):
        model=self.model
        topic_labels=self.topic_labels
        print('\n%s' % author)
        #print('Docs:', model.author2doc[author])
        print('Topics:')
        print([(topic_labels[topic[0]], topic[1]) for topic in model[author]])
        dist=matutils.sparse2full(model[author], model.num_topics)
        df=pd.DataFrame({'Topic':topic_labels,'Score':dist})
        #plt.plot(dist)
#        ax=df['Score'].plot(kind='bar')
#        ax.set_xticklabels(topic_labels, rotation=90)
#        output_file("AuthorTopicDistribution.html")
#        plt.show()
        p = Bar(df, 'Topic', values='Score', title="Bar Plot of Topic Distributions of %s" %self.get_author_name_from_id(author))
        show(p)
        
        print(self.get_author_name_from_id(author))
        
        
    
    def show_author_by_name(self,name):
        Name,Id=self.get_closest_author(name)
        if(Name!='null'):
            if(Name==name):
                self.show_author_by_id(Id)
            else:
                self.show_author_by_id(Id)
                
        
        
        
    def get_author_name_from_id(self,strId):
        return self.df_authors['name'][self.df_authors['id']==strId].values[0]
    
    def get_author_id_from_name(self,strname):
        return self.df_authors['id'][self.df_authors['name']==strname].values[0]


    
    def get_closest_author(self,author_name):
        choices =self.df_authors['name'].values
        names= difflib.get_close_matches(author_name,choices)
        if(len(names)==0):
            print('Couldn\'t find the author or authors with similar names')
            name='null'
            Id=-1
        else:
            name=names[0]
            if(name!=author_name):
                print('\nDidn\'t find any author with the name: %s'%author_name)
                print('\n Found similar name: %s' %name)
                print('\n Showing results for: %s' %name)
                
            
                
            Id=self.get_author_id_from_name(name)
        return name,Id
    
    
    
    
    
    
    def get_author_topic_distribution(self,author_num):
        return matutils.sparse2full(self.author_vecs[author_num], self.model.num_topics)
    
        
            
    def set_topic_labels(self,topic_labels):
        self.topic_labels=topic_labels
    
        
        
    def plot_author_tsne_plot(self,smallest_author=1):
        model=self.model
        tsne=self.tsne
        authors = [model.author2id[a] for a in model.author2id.keys() if len(model.author2doc[a]) >= smallest_author]
        
        # Tell Bokeh to display plots inside the notebook.
        output_file("tsne.html", title="Author TSNE plot")
        x = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        author_ids = [model.id2author[a] for a in authors]
        author_names = [self.get_author_name_from_id(a) for a in author_ids]
        
        
        
        # Radius of each point corresponds to the number of documents attributed to that author.
        scale = 0.1
        author_sizes = [len(model.author2doc[a]) for a in author_ids]
        radii = [size * scale for size in author_sizes]
        
        source = ColumnDataSource(
                data=dict(
                    x=x,
                    y=y,
                    author_names=author_names,
                    author_sizes=author_sizes,
                    radii=radii,
                )
            )
        
        # Add author names and sizes to mouse-over info.
        hover = HoverTool(
                tooltips=[
                ("author", "@author_names"),
                ("size", "@author_sizes"),
                ]
            )
        
        p = figure(tools=[hover, 'crosshair,pan,wheel_zoom,box_zoom,reset,save,lasso_select'])
        p.scatter('x', 'y', radius='radii', source=source, fill_alpha=0.6, line_color=None)
        show(p)
        
    
        

if __name__=='__main__':
    n_topics=9
    Tm=A_TM('ATM'+str(n_topics))
    #Tm.create_AT_model(n_topics)
    Tm.load_existing_model()
    #Tm.plot_author_clustering_interia(max_cluster=20)
    #Tm.create_classification_from_cluster_data();
    #Tm.show_author('13')
    #df=Tm.get_sim_author_table('13')
    #Tm.plot_author_tsne_plot()
    

        
    
        
    
    