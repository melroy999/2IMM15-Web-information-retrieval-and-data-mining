import string

# Used to remove punctuation.
normalization_table = str.maketrans({key: None for key in string.punctuation})


def remove_punctuation(text):
    # Remove all punctuation.
    return text.translate(normalization_table)


def normalize(terms):
    return [normalize_word(term) for term in terms]


def normalize_word(term):
    return term.translate(normalization_table).lower()
