# TODO Term normalisation
# TODO Mapping from terms to documents in which the term occurs
# TODO For each document, add indices for term occurrence
import string
from collections import defaultdict, OrderedDict

import data as data

normalization_table = str.maketrans({key: None for key in string.punctuation})

def tokenize(content):
    # return [word.strip(string.punctuation) for word in content.split()]
    return content.split()


def normalize(term):
    # First remove all punctuation.
    term = term.translate(normalization_table)

    # Now make it lower case.
    return term.lower()


# Import the list of papers and authors.
papers = data.import_papers()
authors = data.import_authors()

# We will keep a dictionary between terms and postings.
postings = defaultdict(dict)

# Walk over all the papers and create the index tables.
for paper in papers:
    print(paper.id)

    # Tokenize the contents of the papers, and normalize the terms.
    tokens = [normalize(term) for term in tokenize(paper.paper_text)]

    # Update the list of postings by adding the id of the paper.
    for i in range(0, len(tokens)):
        term = tokens[i]

        # Skip if it is empty.
        if term == '':
            continue

        # get the dict by key, having a default value of an empty dict.
        position_mapping = postings.get(term, defaultdict(list))

        # Now, for each of the terms, append the positional index of the term.
        position_mapping[paper.id].append(i)

        # Add the dict to the postings mapping.
        postings[term] = position_mapping

# Created a dictionary in which the keys are sorted alphabetically.
ordered_postings = OrderedDict(sorted(postings.items(), key=lambda t : t[0]))