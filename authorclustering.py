import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

from sklearn.cluster import KMeans
import numpy as np
import pickle
import gensim
from gensim import matutils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lda

nips=gensim.models.LdaModel.load('nips.lda')
corpus=pickle.load(open("corpus.p","rb"))
doc_vecs = lda.load_doc_vecs()
X=[matutils.sparse2full(doc, nips.num_topics) for doc in doc_vecs]
inertianew=[]
#scaler = StandardScaler()
#scaler.fit(X)
#X_new=scaler.transform(X)
X_new=X

num=80
for i in range(3,num):
    
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X_new)
    inertianew.append(kmeans.inertia_)

plt.plot(list(range(3,num)),inertianew)
plt.show()
