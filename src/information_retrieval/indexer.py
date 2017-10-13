from collections import Counter, defaultdict
from functools import partial
from multiprocessing.pool import Pool

import time
import math
import gc
import numpy
import pickle

from import_data import database

# The fields we target in the papers.
from information_retrieval.normalizer import Normalizer

# The fields we may target during the indexing.
paper_fields = ["title", "abstract", "paper_text"]

# Define for which fields we want to use multiprocessing.
multiprocessing_table = {
    "title": False,
    "abstract": False,
    "paper_text": True
}


def create_table_file(filename, data):
    with open(filename + ".pickle", "wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_table_file(filename):
    with open(filename + ".pickle", "rb") as input_file:
        print("Pre-calculated data file found!")
        return pickle.load(input_file)


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

    @staticmethod
    def normalize_and_tokenize(text, normalizer):
        # Remove punctuation and convert to lowercase.
        text = normalizer.remove_punctuation(text.lower())

        # Removing control characters takes quite long, so do it last, when we already have less characters.
        text = normalizer.remove_control_characters(text)

        # Create tokens.
        return text.split()

    # Calculate the term frequencies of the tokens, with the given normalizer.
    @staticmethod
    def calculate_tf(tokens, normalizer):
        tf = defaultdict(int)
        for term, value in Counter(tokens).items():
            if normalizer.is_valid_term(term):
                norm_term = normalizer.normalize(term)
                tf[norm_term] = tf[norm_term] + value
        return tf

    # Process the term frequencies of the given text.
    @staticmethod
    def process_text(text, normalizer):
        # Dictionary that will hold all data we know about this papers text.
        data = {}

        # Normalize and convert the text to tokens.
        tokens = Indexer.normalize_and_tokenize(text, normalizer)

        # Generate the tf frequencies, with stemming and stopwords.
        tf = Indexer.calculate_tf(tokens, normalizer)

        # Calculate the wf measure.
        # noinspection PyArgumentList
        wf = defaultdict(int, {term: 1 + math.log10(value) for term, value in tf.items()})

        # Store the frequency under the key frequencies in the data.
        data["tf"] = tf
        data["wf"] = wf

        # Calculate the vector lengths of the tf and wf measures, and store it under the key vector_lengths.
        data["vector_lengths"] = {
            "tf": math.sqrt(sum([x**2 for x in tf.values()])),
            "wf": math.sqrt(sum([x**2 for x in wf.values()])),
        }

        # Also store the other information we have about the paper, such as the #terms and #unique terms.
        data["number_of_terms"] = sum(tf.values())
        data["number_of_unique_terms"] = len(tf)

        # Return the data dictionary containing all information about the text.
        return data

    # Process the term frequencies of the terms found in the paper.
    @staticmethod
    def process_paper(paper, field, normalizer):
        # Choose the text, and use process_text.
        data = Indexer.process_text(paper.__getattribute__(field), normalizer)

        # Having the id in there is useful for debugging purposes.
        data["paper_id"] = paper.id
        return data

    # Calculate the initial data we can gather from the papers for the given field.
    def calculate_papers_data(self, papers, field, normalizer):
        # The data we will return.
        data = {}

        # First gather the tf and wf measures of the papers.
        self.update_status("Indexing field \"" + field + "\"... gathering term frequencies...")
        if multiprocessing_table[field]:
            with Pool(4) as pool:
                # Schedule the papers to be processed by the pool, and save the term frequency data.
                papers_frequency_data = pool.map(partial(self.process_paper, field=field, normalizer=normalizer),
                                                 papers)
                data = {papers[i].id: value for i, value in enumerate(papers_frequency_data)}
        else:
            for paper in papers:
                data[paper.id] = self.process_paper(paper, field, normalizer)

        # Return the data...
        return data

    # Calculate the collection frequencies. I.e, cf, df and idf.
    @staticmethod
    def calculate_collection_data(paper_frequency_data):
        # We have more than just data point in the collection, so we need another dictionary.
        data = {}

        cf = defaultdict(int)
        df = defaultdict(int)
        for paper_frequencies in paper_frequency_data.values():
            for term, value in paper_frequencies["tf"].items():
                cf[term] += value
                df[term] += 1

        # Now, calculate the idf.
        # noinspection PyArgumentList
        idf = defaultdict(int, {term: numpy.log10(len(database.papers) / value) for term, value in df.items()})

        # Assign the frequency data under the key "frequencies"
        data["cf"] = cf
        data["df"] = df
        data["idf"] = idf

        # Calculate the vector lengths and other statistics that might be useful.
        data["vector_lengths"] = {
            "cf": math.sqrt(sum([x**2 for x in cf.values()])),
            "df": math.sqrt(sum([x**2 for x in df.values()])),
            "idf": math.sqrt(sum([x**2 for x in idf.values()])),
        }

        data["number_of_terms"] = sum(cf.values())
        data["number_of_unique_terms"] = len(cf)

        # Return the gathered data.
        return data

    @staticmethod
    def calc_tf_idf_and_wf_idf(idf, values):
        tf_idf = defaultdict(int)
        wf_idf = defaultdict(int)

        tf_idf_length = 0
        for term, value in values["tf"].items():
            val = tf_idf[term] = value * idf[term]
            tf_idf_length += val ** 2

        wf_idf_length = 0
        for term, value in values["wf"].items():
            val = wf_idf[term] = value * idf[term]
            wf_idf_length += val ** 2

        values["tf_idf"] = tf_idf
        values["wf_idf"] = wf_idf

        # Calculate the vector lengths.
        values["vector_lengths"]["tf_idf"] = tf_idf_length
        values["vector_lengths"]["wf_idf"] = wf_idf_length

    @staticmethod
    def calc_tf_idf_and_wf_idf_for_papers(idf, paper_frequency_data):
        # Update the tf idf in papers.
        for paper_id, values in paper_frequency_data.items():
            tf_idf = defaultdict(int)
            wf_idf = defaultdict(int)

            tf_idf_length = 0
            for term, value in values["tf"].items():
                val = tf_idf[term] = value * idf[term]
                tf_idf_length += val ** 2

            wf_idf_length = 0
            for term, value in values["wf"].items():
                val = wf_idf[term] = value * idf[term]
                wf_idf_length += val ** 2

            values["tf_idf"] = tf_idf
            values["wf_idf"] = wf_idf

            # Calculate the vector lengths.
            values["vector_lengths"]["tf_idf"] = tf_idf_length
            values["vector_lengths"]["wf_idf"] = wf_idf_length

    # Index a certain field for all the papers, with multiprocessing when defined.
    def index(self, papers, normalizer):
        # A dictionary that will hold all the data we have for the given field of the given papers.
        data = {"collection": {}, "papers": {}, "N": len(database.papers)}

        # We must do the following for all fields.
        for field in paper_fields:
            # First gather all the data that we can gather from the papers alone.
            papers_data = data["papers"][field] = self.calculate_papers_data(papers, field, normalizer)

            # Now continue with the collection data.
            self.update_status("Indexing field \"" + field + "\"... gathering collection frequencies...")
            data["collection"][field] = self.calculate_collection_data(papers_data)

            # Report on the amount of terms found for the field.
            number_of_unique_terms = data["collection"][field]["number_of_unique_terms"]
            print("Found", number_of_unique_terms, "unique terms for the field \"" + field + "\".")

        # Return the collected data.
        return data

    def reset(self):
        self.results = None
        gc.collect()

    def calculate_missing_measures(self):
        for field in paper_fields:
            idf = self.results["collection"][field]["idf"]

            # Calculate the missing measures afterwards, as we don't want to store them.
            self.calc_tf_idf_and_wf_idf_for_papers(idf, self.results["papers"][field])

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

        # Try to fetch the indexing data from disk.
        try:
            self.results = load_table_file("../../data/calc_" + self.normalizer.operator_name.lower().replace(" ", "_"))
        except (FileNotFoundError, EOFError):
            print("Pre-calculated data file not found. Recalculating... this may take a long time.")

            # Do the indexing.
            self.results = self.index(database.papers, self.normalizer)

            # Create a cheat file which should make normalizations faster next time.
            self.normalizer.create_table_file()

            # Create a cheat file for the entire index data.
            self.write_index_to_disk()

        # We still need to calculate some missing measures that would not have been stored in the table.
        self.calculate_missing_measures()

        # Report the time.
        print()
        print("Finished indexing in", time.time() - start, "seconds.")
        self.update_status("Finished indexing.")
        print()

    def write_index_to_disk(self):
        print("Storing indexing data on disk for later use.")

        normalizer_name = self.normalizer.operator_name.lower().replace(" ", "_")
        filename = "../../data/calc_" + normalizer_name
        create_table_file(filename, self.results)

    # During indexing of a query, do the exact same to the query as would be done with the paper contents.
    def index_query(self, query):
        # Call the index text directly for the query.
        return self.process_text(query, self.normalizer)

    # Initializes the indexer.
    def __init__(self):
        # Load the papers.
        database.import_papers()


if __name__ == "__main__":
    indexer = Indexer()
    indexer.full_index("None", True, None)
    f = indexer.results

    print(len(f["papers"]["paper_text"][1]["tf"]))
    print(len(f["papers"]["paper_text"][1]["tf_idf"]))
    print(f["papers"]["paper_text"][1]["vector_lengths"])
