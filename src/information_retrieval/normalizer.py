import pickle
import string

from nltk.stem.wordnet import WordNetLemmatizer

# Different lemmatizers we can use.
nltk_wordnet_lemmatizer = WordNetLemmatizer().lemmatize

from nltk.stem import *

# Different stemmings we can use.
nltk_porter_stemmer = PorterStemmer().stem
nltk_lancaster_stemmer = LancasterStemmer().stem
nltk_snowball_stemmer = SnowballStemmer("english").stem

# A table that will contain punctuation to be removed.
punctuation_removal_table = str.maketrans({key: None for key in string.punctuation})


# A stemmer that just returns the original term.
def no_stemming(term):
    return term

name_to_normalizer = {
    "Nltk wordnet lemmatizer": nltk_wordnet_lemmatizer,
    "Nltk porter stemmer": nltk_porter_stemmer,
    "Nltk lancaster stemmer": nltk_lancaster_stemmer,
    "Nltk snowball stemmer": nltk_snowball_stemmer,
    "None": no_stemming
}


# A dict that just returns the key that was requested.
class CaptainObviousDict(dict):
    def __missing__(self, key):
        return key


class Normalizer:
    def __init__(self, use_stopwords, operator_name=list(name_to_normalizer.keys())[0]):
        self.use_stopwords = use_stopwords
        self.operator_name = operator_name
        self.operator = name_to_normalizer[operator_name]

        # A dictionary that will hold already stemmed words.
        try:
            if self.operator_name == "None":
                self.term_to_normalized_term = CaptainObviousDict()
            else:
                self.term_to_normalized_term = self.load_table_file()
                print("Successfully loaded previously saved \"" + operator_name + "\" mapping table.")
                print()
            self.skip_file_write = True
        except FileNotFoundError:
            self.term_to_normalized_term = {}
            self.skip_file_write = False

    # Stem the words, using the lambda expression as the stemmer / lemmatizer.
    def normalize(self, term):
        try:
            return self.term_to_normalized_term[term]
        except KeyError:
            self.term_to_normalized_term[term] = self.operator(term)
            return self.term_to_normalized_term[term]

    # Dump the term to normalized term table for later use.
    def create_table_file(self):
        if not self.skip_file_write:
            with open("../../data/" + self.operator_name + ".pickle", "wb") as output_file:
                pickle.dump(self.term_to_normalized_term, output_file)

    # Load the term to normalized term table from disk.
    def load_table_file(self):
        with open("../../data/" + self.operator_name + ".pickle", "rb") as input_file:
            return pickle.load(input_file)

    # Remove punctuation in the given text.
    @staticmethod
    def remove_punctuation(text):
        # Remove all punctuation.
        return text.translate(punctuation_removal_table)

    # Check whether the term is a stop word.
    def is_valid_term(self, term):
        return not self.use_stopwords or not english_stopwords.__contains__(term)


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
