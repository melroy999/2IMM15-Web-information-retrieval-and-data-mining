from nltk.stem.wordnet import WordNetLemmatizer

# Different lemmatizers we can use.
_nltk_wordnet_lemmatizer_object = WordNetLemmatizer()
nltk_wordnet_lemmatizer = _nltk_wordnet_lemmatizer_object.lemmatize

# A dictionary that will hold already stemmed words.
term_to_lemma = {}


# Lemmatizing using the nltk wordnet lemmatizer.
def execute_nltk_wordnet_lemmatizer(term):
    return lemmatize(term, lambda x: nltk_wordnet_lemmatizer(x))


# Stem the words, using the lambda expression as the stemmer / lemmatizer.
def lemmatize(term, lambda_expression):
    try:
        return term_to_lemma[term]
    except KeyError:
        term_to_lemma[term] = lambda_expression(term)
        return term_to_lemma[term]