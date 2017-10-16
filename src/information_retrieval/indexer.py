import gc

from import_data import database

# The fields we target in the papers.
from information_retrieval.normalizer import Normalizer

# The fields we may target during the indexing.
paper_fields = ["title", "abstract", "paper_text"]


class Indexer(object):
    # Holds the results of the indexing.
    results = None

    # Holds the normalizer last used in this indexer.
    normalizer = None

    def reset(self):
        self.results = None
        gc.collect()

    # Initializes the indexer.
    def __init__(self):
        # Load the papers and authors.
        database.import_data()


if __name__ == "__main__":
    indexer = Indexer()
    f = indexer.results

    print(len(f["papers"]["paper_text"][1]["tf"]))
    print(len(f["papers"]["paper_text"][1]["tf_idf"]))
    print(f["papers"]["paper_text"][1]["vector_lengths"])
