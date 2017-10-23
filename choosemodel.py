import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk


#import LDA_TM
#p=[]
#training_acc=[]
#val_acc=[]
#for i in range(4,25):
#    Test=LDA_TM.LDA_TM('LDA'+str(i+1))
#    #Test.create_LDA_model(i+1)
#    #p.append(Test.per_word_bound)
#    Test.load_LDA_model(Test.model_name)
#        
#    Test.create_doc_vecs()
#        
#    Test.load_existing_model()
#    Test.create_doc_clustering(i+1)
#    hist=Test.create_classification_from_cluster_data();
#    training_acc.append(hist.history['acc'][-1])
#    val_acc.append(hist.history['val_acc'][-1])
#    
#    
#    
#    
x=[i+1 for i in range(4,25)]
plt.plot(x,training_acc,label='Training Accuracy')
plt.plot(x,val_acc,label='Validation Accuracy')
plt.xlabel('Number of topics(=number of classes=number of clusters)')
plt.legend()
plt.title('Variation in training and test accuracy of document classification with increase in number of topics')
plt.show()

