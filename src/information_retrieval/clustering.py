import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from import_data import database
import cleanup_module.cleanup

from sklearn.feature_extraction import DictVectorizer
from information_retrieval.indexer import Indexer

# Import the vectorization as done by the indexing.
indexer = Indexer()
indexer.index_corpus("Nltk porter stemmer", True)

# Use the results found in the indexing as the vector.
vectorizer = DictVectorizer()
X = vectorizer.fit_transform([result["tf.idf"] for result in indexer.results["papers"]["paper_text"].values()])

# cleanup_module.cleanup.get_cleanup_instance(database).clean(database.papers)
#
# documents = [paper.paper_text for paper in database.papers]
#
# # Vectorize the text i.e. convert the strings to numeric features
# start = time.time()
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents)
# print(time.time() - start)

# Cluster documents
start = time.time()
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit_transform(X)

# Print top terms per cluster clusters
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("Cluster %d:" % i, '\n')
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)
    print ('\n')
print(time.time() - start)


##########################################################################################
# Visualizing section
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Convert to 2 dimensions using truncaded SVD as we have a sparse matrix and PCA does not work with it
tSVD = TruncatedSVD(n_components=2).fit(X)
data2D = tSVD.transform(X)

# Plot graph with all nodes
plt.scatter(data2D[:,0], data2D[:,1], s=2, c= model.labels_)

centers2D = tSVD.transform(model.cluster_centers_)

# Plot centroids
plt.scatter(centers2D[:,0], centers2D[:,1], marker='x', s=200, linewidths=3, c='r')

plt.show()
