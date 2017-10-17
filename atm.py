import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import AuthorTopicModel
import pickle
import gensim
from gensim import matutils


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

def load_author_vecs(fname="author_vecs.p"):
    print('\nLoading author_vecs: ')
    author_vecs=pickle.load(open(fname,"rb"))
    print("Loading Completed")
    return author_vecs


def load_model(name='model.atmodel'):
    model=AuthorTopicModel.load(name)
    return model
    
    


def get_vector_similarity_hellinger(vec1, vec2,model):
    '''Get similarity between two vectors'''
    
    dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics), \
                              matutils.sparse2full(vec2, model.num_topics))
    sim = 1.0 / (1.0 + dist)
    return sim

def get_sims(vec,vecs,model='load'):
    ''''Get similarity between a vector and a collection of vectors'''
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    
    sims = [get_vector_similarity_hellinger(vec, vec2,model) for vec2 in vecs]
    return sims

def get_sim_author_table(name, top_n=10,model='load',corpus='load',author_vecs='load',smallest_author=1):
    
    if(corpus=='load'):
        corpus=load_corpus()
        
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    
    if(author_vecs=='load'):
        print('Loading saved author_vecs from author_vecs.p')
        author_vecs=load_author_vecs()
    
    sims = get_sims(model.get_author_topics(name),author_vecs,model)

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = model.id2author[elem[0]]
        sim = elem[1]
        author_size = len(model.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))
            
    # Make dataframe and retrieve top authors.
    df = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    df = df.sort_values('Score', ascending=False)[:top_n]
    
    return df


def create_author_vecs(model='load',corpus='load'):
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    if(corpus=='load'):
        corpus=load_corpus()
    
    
    author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]
    pickle.dump(author_vecs,open("author_vecs.p","wb"))
    print('\nSaving author_vecs')
    
    
    

def printtopics(topic_labels='not_assigned',model='load'):
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i) for i in range(model.num_topics)]
    for topic in model.show_topics(num_topics=model.num_topics):
        print('Label: ' + topic_labels[topic[0]])
        words = ''
        for word, prob in model.show_topic(topic[0]):
            words += word + ' '
        print('Words: ' + words)
        print()


def print_top_titles_by_topic(model='load',topic_labels='not_assigned'):
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i) for i in range(model.num_topics)]
    
    
    
    



def create_ATM_model(num_topics,corpus='load',dictionary='load',author2doc='load'):
    if(corpus=='load'):
        corpus=load_corpus()
    if(dictionary=='load'):
        print('Loading saved dictionary from dic.p')
        dictionary=load_dictionary()
    if(author2doc=='load'):
       author2doc=pickle.load(open("author2docModel.p","rb"))
       print("Loading Completed")

    model_list = []
    for i in range(5):
        model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=dictionary.id2token, \
                        author2doc=author2doc, chunksize=2000, passes=100, gamma_threshold=1e-10, \
                        eval_every=0, iterations=1, random_state=i)
        top_topics = model.top_topics(corpus)
        tc = sum([t[1] for t in top_topics])
        model_list.append((model, tc))
    model, tc = max(model_list, key=lambda x: x[1])
    print('Topic coherence: %.3e' %tc)
    model.save('model.atmodel')
    print('AT Model saved as model.atmodel')
    print('Creating author Vecs')
    create_author_vecs(model,corpus)
    
    
    
def show_author(author,model,df_authors,topic_labels='not_assigned'):
    if(model=='load'):
        print('Loading saved model from model.atmodel')
        model=load_model()
    if(topic_labels=='not_assigned'):
        topic_labels=['Topic #'+str(i+1) for i in range(model.num_topics)]
    
    print('\n%s' % author)
    print('Docs:', model.author2doc[author])
    print('Topics:')
    print([(topic_labels[topic[0]], topic[1]) for topic in model[author]])
    dist=matutils.sparse2full(model[author], model.num_topics)
    df=pd.DataFrame({'Topic':topic_labels,'Score':dist})
    #plt.plot(dist)
    ax=df['Score'].plot(kind='bar')
    ax.set_xticklabels(topic_labels, rotation=90)
    plt.show()
    print(get_name_from_id(author,df_authors))
    
def get_name_from_id(strId,df_authors):
    return df_authors['name'][df_authors['id']==strId].values[0]
    
    
if __name__=='__main__':
    num_topics=8
    #printtopics()
    #create_author_vecs(load_model(),load_corpus())
    #print(get_sim_author_table('13'))
    show_author('2',model,df_authors)
    
    
    


    

