import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
#import LDA_TM
#import A_TM
from TopicModeling import LDA_TM
from TopicModeling import A_TM


class TM():
    def __init__(self,num_topics):
        self.n_topics=num_topics
        self.lda=LDA_TM.LDA_TM('LDA'+str(num_topics))
        self.atm=A_TM.A_TM('ATM'+str(num_topics))
    
    def load_existing_model(self):
        self.lda.load_existing_model()
        self.atm.load_existing_model()
    
    def create_model(self):
        self.lda.create_LDA_model(self.n_topics)
        self.atm.create_AT_model(self.n_topics)
    
    def get_author_info(self,author_name):
        #self.atm.show_author_by_name(author_name)
        name,Id=self.atm.get_closest_author(author_name)
        print('\n\n\n Following are the IDs of Docs Published by author:')
        print(self.lda.get_author_docs(Id))
        print('\n\n\n Getting Similar authors for %s:'%name)
        print(self.atm.get_sim_author_table(Id))
        print('\n\n\n Getting Evolution plot of %s for every year:'%name)
        df=self.lda.plot_author_evolution_plot(Id)
        print('\n\n\n Getting Top topic plot of %s for every year :'%name)
        self.lda.plot_top_topic_of_author_by_year(df)
    
    def get_doc_info(self,docId,top_n=10):
        print(self.lda.get_doc_topic_distribution_by_id(docId))
        print('Fetching Similar Documents by measuring Hellinger Distance:')
        print(self.lda.get_doc_sim_table_from_doc_id(docId,top_n))
        

if __name__=='__main__':
    #import LDA_TM
    #import A_TM
    n_topics=9
    Tm=TM(n_topics)
    #Tm.create_model()
    Tm.load_existing_model()
    Tm.get_author_info('M Jordan')
    




        
        
        
        
        
        