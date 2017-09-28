import string

import data

# Used to remove punctuation.
normalization_table = str.maketrans({key: None for key in string.punctuation})


def tokenize(content):
    return content.split()


def normalize(term):
    # First remove all punctuation.
    term = term.translate(normalization_table)

    # Now make it lower case.
    return term.lower()


# Import the list of papers and authors.
papers = data.import_papers()
authors = data.import_authors()

