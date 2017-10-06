from collections import Counter, defaultdict
from functools import partial
from multiprocessing.pool import Pool

import time

import math

import gc

from import_data import database

# The fields we target in the papers.
from information_retrieval.normalizer import Normalizer

# The fields we may target during the indexing.
paper_fields = ["title", "abstract", "paper_text"]


class Indexer(object):
    # The status bar reference.
    status_bar = None

    # Holds the results of the indexing.
    results = None

    # Holds the normalizer last used in this indexer.
    normalizer = None

    # Update the status in the gui, if possible.
    def update_status(self, status):
        try:
            self.status_bar(status)
        except TypeError:
            pass

    # Process the term frequencies of the given text.
    @staticmethod
    def process_text(text, normalizer):
        lowercase = text.lower()
        no_punctuation = normalizer.remove_punctuation(lowercase)
        tokens = no_punctuation.split()

        # Generate the tf frequencies, with stemming and stopwords.
        tf = Indexer.calculate_tf(tokens, normalizer)

        # Generate the wf frequencies and calculate the length of the two vectors.
        tf_length, wf, wf_length = Indexer.calculate_wf_and_lengths(tf)

        return tf, wf, math.sqrt(tf_length), math.sqrt(wf_length)

    # Process the term frequencies of the terms found in the paper.
    @staticmethod
    def process_paper(paper, field, normalizer):
        # Choose the text, and use process_text.
        return Indexer.process_text(paper.__getattribute__(field), normalizer)

    # Calculate the term frequencies of the tokens, with the given normalizer.
    @staticmethod
    def calculate_tf(tokens, normalizer):
        tf = defaultdict(int)
        for term, value in Counter(tokens).items():
            if normalizer.is_valid_term(term):
                norm_term = normalizer.normalize(term)
                tf[norm_term] = tf[norm_term] + value
        return tf

    # Calculate the weighted term frequencies, next to the document vector lengths.
    @staticmethod
    def calculate_wf_and_lengths(term_frequencies):
        tf_length = 0
        wf_length = 0
        wf = defaultdict(int)
        for term, value in term_frequencies.items():
            wf[term] = 1 + math.log2(value)
            tf_length += value ** 2
            wf_length += wf[term] ** 2
        return tf_length, wf, wf_length

    # Calculate the tf_idf and wf_idf measures, and calculate their lengths.
    @staticmethod
    def calculate_tf_idf(vector, idf_collection):
        tf, wf, tf_length, wf_length = vector
        tf_idf = defaultdict(int)
        tf_idf_length = 0
        for term, value in tf.items():
            value = tf_idf[term] = value * idf_collection[term][2]
            tf_idf_length += value ** 2
        tf_idf_length = math.sqrt(tf_idf_length)

        wf_idf = defaultdict(int)
        wf_idf_length = 0
        for term, value in wf.items():
            value = wf_idf[term] = value * idf_collection[term][2]
            wf_idf_length += value ** 2
        wf_idf_length = math.sqrt(wf_idf_length)

        return tf, wf, tf_idf, wf_idf, tf_length, wf_length, tf_idf_length, wf_idf_length

    # Index a certain field for all the papers, with multiprocessing when defined.
    def index(self, papers, field, normalizer, multiprocessing=True):
        self.update_status("Indexing field \"" + field + "\"... gathering term frequencies...")
        if multiprocessing:
            with Pool(4) as pool:
                # Schedule the papers to be processed by the pool, and save the term frequency data.
                paper_tfs = pool.map(partial(self.process_paper, field=field, normalizer=normalizer), papers)
        else:
            paper_tfs = []
            for paper in papers:
                paper_tfs.append(self.process_paper(paper, field, normalizer))

        # Calculate the collective statistics, such as the cf and df measures.
        self.update_status("Indexing field \"" + field + "\"... gathering collection frequencies...")
        idf_collection = self.calculate_cf_df(paper_tfs)

        # Calculate the inverse document frequency.
        self.update_status("Indexing field \"" + field + "\"... calculating inverted document frequency...")
        idf_collection, idf_collection_length = self.calculate_idf_and_idf_length(idf_collection)

        # Pre-calculate tf.idf and wf.idf.
        self.update_status("Indexing field \"" + field + "\"... calculating and normalizing tf.idf + wf.idf ...")
        paper_frequencies = []
        for vector in paper_tfs:
            paper_frequencies.append(self.calculate_tf_idf(vector, idf_collection))

        # Report on the amount of terms found for the field.
        print("Found", len(idf_collection), "unique terms for the field \"" + field + "\".")

        return paper_frequencies, idf_collection, idf_collection_length

    # Calculate the collection frequency and the document frequency.
    @staticmethod
    def calculate_cf_df(paper_term_frequencies):
        idf_collection = defaultdict(lambda: (0, 0, 0))
        for paper_tf, _, _, _ in paper_term_frequencies:
            for term, value in paper_tf.items():
                x, y, _ = idf_collection[term]
                idf_collection[term] = (x + value, y + 1, 0)

        return idf_collection

    # Calculate the inverse document frequency and the idf document vector length.
    @staticmethod
    def calculate_idf_and_idf_length(idf_collection):
        paper_log = math.log2(len(database.papers))
        idf_collection_length = 0
        for term, (cf, df, _) in idf_collection.items():
            idf_collection[term] = (cf, df, paper_log - math.log2(df))
            idf_collection_length += idf_collection[term][2] ** 2
        return idf_collection, idf_collection_length

    def reset(self):
        self.results = None
        gc.collect()

    # Calculate a full index of the papers.
    # This includes the fields: paper_text, abstract, title
    def full_index(self, normalizer_name, use_stopwords, status_bar):
        # Reset to avoid memory issues!
        self.reset()

        # Set the status update bar update function.
        self.status_bar = status_bar

        # Start a timer for performance measures.
        start = time.time()

        # Create a normalizer object.
        self.normalizer = Normalizer(use_stopwords, normalizer_name)

        # Index the different fields of the paper.
        self.results = {
            "paper_text": self.index(database.papers, "paper_text", self.normalizer, True),
            "abstract": self.index(database.papers, "abstract", self.normalizer, False),
            "title": self.index(database.papers, "title", self.normalizer, False)
        }

        # Report the time.
        print()
        print("Finished indexing in", time.time() - start, "seconds.")
        self.update_status("Finished indexing.")
        print()

        # Create a cheat file which should make normalizations faster next time.
        self.normalizer.create_table_file()

    # During indexing of a query, do the exact same to the query as would be done with the paper contents.
    def index_query(self, query):
        # Call the index text directly for the query.
        return self.process_text(query, self.normalizer)

    # Initializes the indexer.
    def __init__(self):
        # Load the papers.
        database.import_papers()
