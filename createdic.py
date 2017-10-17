# Create a dictionary representation of the documents, and filter out frequent and rare words.

from gensim.corpora import Dictionary
import pickle
dictionary = Dictionary(d)
print(len(dictionary))

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.

pickle.dump(dictionary,open("dic.p","wb"))
