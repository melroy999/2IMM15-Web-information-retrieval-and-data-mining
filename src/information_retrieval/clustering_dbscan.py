from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score

from sklearn.feature_extraction import DictVectorizer
from information_retrieval.indexer import Indexer

# Import the vectorization as done by the indexing.
indexer = Indexer()
indexer.index_corpus("Nltk porter stemmer", True)
# Get the normalized scores for the papers
normalized = indexer.get_normalized_paper_values("paper_text", "wf.idf")

# Use the results found in the indexing as the vector.
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(normalized.values())

db = DBSCAN(metric="precomputed").fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))

