from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Need the Tf-idf-weighted document-term matrix to use KMeans
from information_retrieval.indexer import Indexer, paper_fields

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

#vectorize the text i.e. convert the strings to numeric features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

#cluster documents
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