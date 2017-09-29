# TODO Term normalisation
# TODO Mapping from terms to documents in which the term occurs
# TODO For each document, add indices for term occurrence

from collections import defaultdict, Counter
from math import log10
from string import punctuation
from time import time

from retrieval.data import import_papers, import_authors

normalization_table = str.maketrans({key: None for key in punctuation})


def tokenize(content):
    # return [word.strip(string.punctuation) for word in content.split()]
    return content.split()

def normalize(content):
    # Remove all special symbols.
    terms = [term.translate(normalization_table) for term in tokens]

    # Now make it lower case.
    return map(str.lower, terms)


# Import the list of papers and authors.
papers = import_papers()
paper_ids = [paper.id for paper in papers]
authors = import_authors()
author_ids = [author.id for author in authors]

# Performance analysis counter.
start = time()

# Print the time taken.
print("Setup: ", time() - start)

# Performance analysis counter.
start = time()

# The data we gather about terms.
data = {}


class TermIndex:
    def __init__(self):
        self.cf = 0
        self.df = 0
        self.idf = 0
        self.papers_tf = defaultdict(int)
        self.papers_wf = defaultdict(int)


# Walk over all the papers and create the information_retrieval tables.
for paper in papers:
    # Tokenize the paper.
    tokens = tokenize(paper.paper_text)

    # Normalize the terms.
    terms = normalize(tokens)

    # Count the amount of tokens.
    counter = Counter(terms)

    # Now, use the values in the counter to update the data.
    for term in counter.keys():
        # Create the data item if it does not exist yet.
        if term not in data:
            data[term] = TermIndex()

        # Update the collection frequency.
        data[term].cf += counter[term]

        # Increment the document frequency.
        data[term].df += 1

        # Update the paper frequency.
        data[term].papers_tf[paper.id] = counter[term]
        data[term].papers_wf[paper.id] = 1 + log10(counter[term])


# Print the time taken.
print("Running: ", time() - start)
print("The frequency: " + str(data["the"].cf))
print("Of frequency: " + str(data["of"].cf))
print("The frequency paper 1: " + str(data["the"].papers_tf[1]))
print("Of frequency paper 1: " + str(data["of"].papers_tf[1]))