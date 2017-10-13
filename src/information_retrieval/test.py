import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import import_data.database as database


database.import_papers()
papers = database.papers

text = [paper.paper_text for paper in papers]

start = time.time()
tf_data = CountVectorizer().fit_transform(text)
tf_idf_data = TfidfVectorizer().fit_transform(text)
print(time.time() - start)

