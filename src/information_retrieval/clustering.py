from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from information_retrieval.indexer import Indexer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def clusterKMeans(stemmer, function, clusters, seeds):
    # First, set up the data correctly
    # Import the vectorization as done by the indexing.
    indexer = Indexer()
    indexer.index_corpus(stemmer, True)
    normalized = indexer.get_normalized_paper_values("paper_text", function)

    # Use the results found in the indexing as the vector.
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(normalized.values())

    # Cluster documents
    model = KMeans(n_clusters=clusters, init='k-means++', n_init=seeds)
    model.fit_transform(X)

    # Print top terms per cluster clusters
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    labels, counts = np.unique(model.labels_[model.labels_>=0], return_counts=True)

    for i in range(clusters):
        print("Cluster %d:" % i, '\n')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], )
        print('\n')

    for i in range(clusters):
        print(labels[i], counts[i])

    sil_coeff = silhouette_score(X, model.labels_, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(clusters, sil_coeff))

    return X, model


def clutersgraph(X, model):
    # Visualize the results
    # Convert to 2 dimensions using truncaded SVD as we have a sparse matrix and PCA does not work with it
    tSVD = TruncatedSVD(n_components=2).fit(X)
    data2D = tSVD.transform(X)

    fig = plt.figure(1)
    # Plot graph with all nodes
    plt.scatter(data2D[:,0], data2D[:,1], s=2, c= model.labels_)

    centers2D = tSVD.transform(model.cluster_centers_)

    # Plot centroids
    plt.scatter(centers2D[:,0], centers2D[:,1], marker='x', s=20, linewidths=3, c='r')
    return fig


##########################################################################################
# Evaluation
# from sklearn.metrics import silhouette_score
#
# for n_cluster in range(2, 50):
#     kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1).fit(X)
#     label = kmeans.labels_
#     sil_coeff = silhouette_score(X, label, metric='euclidean')
#     print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

# The scoring measures the user can choose from.
scoring_measures = ["tf.idf", "wf.idf"]

# Choices for stemmer
stemmer = ["Nltk wordnet lemmatizer", "Nltk porter stemmer",
           "Nltk lancaster stemmer", "Nltk snowball stemmer", "None"]