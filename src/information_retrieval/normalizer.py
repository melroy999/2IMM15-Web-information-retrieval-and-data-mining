import json
from nltk.stem import *

# Different lemmatizers we can use.
nltk_wordnet_lemmatizer = WordNetLemmatizer().lemmatize

# Different stemmings we can use.
nltk_porter_stemmer = PorterStemmer().stem
nltk_lancaster_stemmer = LancasterStemmer().stem
nltk_snowball_stemmer = SnowballStemmer("english").stem

# A mapping from readable normalizer name to normalizer object.
name_to_normalizer = {
    "Nltk wordnet lemmatizer": nltk_wordnet_lemmatizer,
    "Nltk porter stemmer": nltk_porter_stemmer,
    "Nltk lancaster stemmer": nltk_lancaster_stemmer,
    "Nltk snowball stemmer": nltk_snowball_stemmer,
    "None": None
}


# A dict that just returns the key that was requested.
class ReturnToSenderDict(dict):
    def __missing__(self, key):
        return key


class Normalizer:
    def __init__(self, normalizer_name, use_stopwords):
        self.use_stopwords = use_stopwords
        self.operator = name_to_normalizer[normalizer_name]
        self.normalizer_name = normalizer_name.lower().replace(" ", "_")

        # We want to keep track of a list of normalizations that have already been executed.
        self.export_file = False
        if self.operator is None:
            self.term_to_normalized_term = ReturnToSenderDict()
        else:
            try:
                self.term_to_normalized_term = self.load_table_file()
            except FileNotFoundError:
                self.term_to_normalized_term = {}
                self.export_file = True

    # Dump the term to normalized term table for later use.
    def create_table_file(self):
        if self.export_file:
            with open("../../data/normalization_" + self.normalizer_name + ".json", "w") as output_file:
                json.dump(self.term_to_normalized_term, output_file)

    # Load the term to normalized term table from disk.
    def load_table_file(self):
        with open("../../data/normalization_" + self.normalizer_name + ".json", "r") as input_file:
            return json.load(input_file)

    # Stem the words, using the lambda expression as the stemmer / lemmatizer. Also eliminate stopwords.
    def normalize(self, term):
        try:
            return self.term_to_normalized_term[term]
        except KeyError:
            v = self.term_to_normalized_term[term] = self.operator(term)
            return v

    # Check if a term is a valid term. I.e, should we observe it or not.
    def is_valid_term(self, term):
        # All terms are valid, unless we use stopwords. In that case the stopwords are invalid.
        if not self.use_stopwords:
            return True
        else:
            return term not in english_stopwords


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
