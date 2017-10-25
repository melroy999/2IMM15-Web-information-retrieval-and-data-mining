from nltk.collocations import *
import nltk
from nltk.corpus import stopwords

l=len(Tm.df_papers['abstract'].values)
abstract_combined=''
for i in range(l):
    abstract_combined=abstract_combined+' '+Tm.df_papers['abstract'].values[i]
filtered_words = [word for word in abstract_combined if word not in stopwords.words('english')]
#finder1 = BigramCollocationFinder.from_words(abstract_combined.split(), window_size = 3)
#finder1.apply_freq_filter(5)
#bigram_measures = nltk.collocations.BigramAssocMeasures()
#for k,v in finder1.ngram_fd.items():
#  print(k,v)
  
  