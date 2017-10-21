from information_retrieval.indexer import Indexer
from sklearn.svm import LinearSVC
import operator

indexer = Indexer(None)
indexer.index_corpus("None", True)

results = indexer.results["papers"]["paper_text"][2]["tf"]
print(results.get("memory"))
#sorted_results = sorted(results.items(), key=operator.itemgetter(1))
#print(sorted_results)

#print(len(f["papers"]["paper_text"][1]["tf"]))
#print(len(f["papers"]["paper_text"][1]["tf_idf"]))
#print(f["papers"]["paper_text"][1]["vector_lengths"])

X = [[1, 2, 3, 4],
     [4, 2, 3, 1],
     [1, 1, 1, 1],
     [4, 4, 2, 1]]
y = [1, 2, 3, 4]
clf = LinearSVC(random_state=0)
clf.fit(X, y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[4, 4, 2, 1]]))