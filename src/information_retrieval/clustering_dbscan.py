from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from information_retrieval.indexer import Indexer

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Import the vectorization as done by the indexing.
indexer = Indexer()
indexer.index_corpus("Nltk porter stemmer", True)

# Get the normalized scores for the papers
normalized = indexer.get_normalized_paper_values("paper_text", "wf.idf")

# Use the results found in the indexing as the vector.
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(normalized.values())

db = DBSCAN(eps=1.5, min_samples=100).fit(X)
labels = db.labels_

print(set(labels))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Number of cluster gotten from the data
print('Estimated number of clusters: %d' % n_clusters_)

unique_labels, counts = np.unique(db.labels_[db.labels_>=0], return_counts=True)
for i in range(n_clusters_):
    print (unique_labels[i], counts[i])

##########################################################################################
# Visualizing section

# Convert to 2 dimensions using truncaded SVD as we have a sparse matrix and PCA does not work with it
tSVD = TruncatedSVD(n_components=2).fit(X)
data2D = tSVD.transform(X)

# Plot graph with all nodes
plt.scatter(data2D[:,0], data2D[:,1], s=2, c=db.labels_)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()