import string

# Used to remove punctuation.
from information_retrieval.lemmatizer import execute_nltk_wordnet_lemmatizer


# A table that will contain punctuation to be removed.
punctuation_removal_table = str.maketrans({key: None for key in string.punctuation})


# Remove punctuation in the given text.
def pre_normalization(text):
    # Remove all punctuation.
    return text.translate(punctuation_removal_table).lower()


# Normalize the given list of terms.
def post_normalization(terms, normalizer=lambda x: execute_nltk_wordnet_lemmatizer(x), use_stopwords=True):
    return [normalizer(term) for term in terms if use_stopwords and not english_stopwords.__contains__(term)]


# A list of english stop words.
english_stopwords = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and',
                     'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
                     'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has',
                     'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it',
                     'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my',
                     'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own',
                     'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the',
                     'their', 'them', 'then', 'there', 'these', 'they', 'this', 'to', 'too', 'us', 'wants', 'was', 'we',
                     'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would',
                     'yet', 'you', 'your'}
