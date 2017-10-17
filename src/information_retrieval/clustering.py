import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from import_data import database
import cleanup_module.cleanup


# Need the Tf-idf-weighted document-term matrix to use KMeans
from information_retrieval.indexer import Indexer, paper_fields

cleanup_module.cleanup.get_cleanup_instance(database).clean(database.papers)

documents = [paper.paper_text for paper in database.papers]

#vectorize the text i.e. convert the strings to numeric features
start = time.time()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
print(time.time() - start)

#cluster documents
start = time.time()
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit_transform(X)

#print top terms per cluster clusters
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("Cluster %d:" % i, '\n')
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)
    print ('\n')
print(time.time() - start)