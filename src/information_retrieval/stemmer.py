import stemming.porter
import stemming.porter2
from nltk.stem import *

# Different stemmings we can use.
nltk_porter_stemmer = PorterStemmer().stem
nltk_lancaster_stemmer = LancasterStemmer().stem
nltk_snowball_stemmer = SnowballStemmer("english").stem

porter_2_stemmer = stemming.porter2.stem
porter_stemmer = stemming.porter.stem

# A dictionary that will hold already stemmed words.
term_to_stem = {}


# Stemming using the nltk porter stemmer.
def execute_nltk_porter_stemmer(term):
    return stem(term, lambda x: nltk_porter_stemmer(x))


# Stemming using the nltk lancaster stemmer.
def execute_nltk_lancaster_stemmer(term):
    return stem(term, lambda x: nltk_lancaster_stemmer(x))


# Stemming using the nltk snowball stemmer.
def execute_nltk_snowball_stemmer(term):
    return stem(term, lambda x: nltk_snowball_stemmer(x))


# Stemming using porter stemmer.
def execute_porter_stemmer(term):
    return stem(term, lambda x: porter_stemmer(x))


# Stemming using porter 2 stemmer.
def execute_porter_2_stemmer(term):
    return stem(term, lambda x: porter_2_stemmer(x))


# Stem the words, using the lambda expression as the stemmer / lemmatizer.
def stem(term, lambda_expression):
    try:
        return term_to_stem[term]
    except KeyError:
        term_to_stem[term] = lambda_expression(term)
        return term_to_stem[term]
