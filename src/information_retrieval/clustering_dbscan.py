from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


def cluster(indexer, function, eps, min_samples):
    # Import the vectorization as done by the indexing.
    # Get the normalized scores for the papers
    normalized = indexer.get_normalized_paper_values("paper_text", function)

    # Use the results found in the indexing as the vector.
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(normalized.values())

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Number of cluster gotten from the data
    print('Estimated number of clusters: %d' % n_clusters_)

    # Getting and printing the clusters and amount of points in them
    labels, counts = np.unique(db.labels_[db.labels_ >= 0], return_counts=True)
    for i in range(n_clusters_):
        print('Cluster %d has %d points in it.' % (labels[i], counts[i]))

    return X, db, n_clusters_


def clustergraph(X, db, n_clusters):
    # Visualizing section

    # Convert to 2 dimensions using truncaded SVD as we have a sparse matrix and PCA does not work with it
    tSVD = TruncatedSVD(n_components=2).fit(X)
    data2D = tSVD.transform(X)

    fig = plt.figure(1)
    # Plot graph with all nodes
    plt.scatter(data2D[:, 0], data2D[:, 1], s=2, c=db.labels_)

    plt.title('Estimated number of clusters: %d' % n_clusters)
    return fig


# ##########################################################################################
# eps = 1.2
# min_samples = 25
# X, db, n_clusters = cluster("Nltk porter stemmer", "tf.idf", eps, min_samples)
#
# clustergraph(X, db, n_clusters)


# The scoring measures the user can choose from.
scoring_measures = ["tf.idf", "wf.idf"]
