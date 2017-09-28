import math
import pickle
from collections import Counter
from time import time
from json import dump
from index.normalizer import remove_punctuation, normalize
from index.tokenizer import tokenize
from retrieval import data


# The location where index files are stored.
file_store = '../../index/'

# File name prefix.
file_prefix = 'tf_paper_'


class TermStatistics:
    def __init__(self, tf):
        self.tf = tf
        self.wf = 1 + math.log(tf)


class PaperIndex:
    """
    A class that holds all indexing information for a given paper.
    """

    def __init__(self, paper, tf=None):
        self.paper = paper

        if tf is None:
            self._process()
            self._store_index()
        else:
            self.term_stats = {term: TermStatistics(value) for term, value in tf.items()}

    def _process(self):
        # First, remove all punctuation.
        text = remove_punctuation(self.paper.paper_text)

        # Next, tokenize the paper's contents.
        tokens = tokenize(text)

        # Now, do the post processing normalization.
        terms = normalize(tokens)

        # Create the term frequency table and the weighted term frequency table.
        tf = dict(Counter(terms))

        # Calculate all statistics we can gather for the paper.
        self.term_stats = {term: TermStatistics(value) for term, value in tf.items()}

    def _store_index(self):
        with open(file_store + file_prefix + str(self.paper.id) + '.pickle', 'wb') as file:
            pickle.dump({term: value.tf for term, value in self.term_stats.items()}, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Stored file " + str(self.paper.id))

    def _store_index_json(self):
        with open(file_store + file_prefix + str(self.paper.id) + '.json', 'w') as file:
            dump(self.term_stats, file, indent=4)


def _load_index(paper_id):
    with open(file_store + file_prefix + str(paper_id) + '.pickle', 'rb') as file:
        return pickle.load(file)


def generate_index():
    """
    Load the index from disk, or generate it.
    """

    # Import the list of papers and authors.
    papers = data.import_papers()
    authors = data.import_authors()

    # The list of indexed papers.
    indexed_papers = []

    for paper in papers:
        try:
            tf = _load_index(paper.id)
            indexed_papers.append(PaperIndex(paper, tf))
        except (FileNotFoundError, EOFError):
            indexed_papers.append(PaperIndex(paper))

    # Tests...
    print(indexed_papers[0].term_stats["the"].tf)
    print(indexed_papers[0].term_stats["of"].tf)
    print(indexed_papers[1].term_stats["the"].tf)
    print(indexed_papers[1].term_stats["of"].tf)


start = time()

generate_index()

print(time() - start)


