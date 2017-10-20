import gc
import json
from collections import Counter, defaultdict

import time
import math

import pickle

from import_data import database
import cleanup_module.cleanup as cleanup

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

    # Initializes the indexer.
    def __init__(self, status_bar=None):
        # Load the papers and authors using the cleanup module, as we don't want all the gibberish.
        cleanup.get_cleanup_instance(database).clean(database.papers)
        self.lookup_wf_calc = {}
        self.lookup_idf_calc = {}
        self.status_bar = status_bar

    # Update the status in the gui, if possible.
    def update_status(self, status):
        try:
            self.status_bar(status)
        except TypeError:
            pass

    # Reset the results of the indexer.
    def reset(self):
        self.results = None
        gc.collect()

    # Calculate the term frequency of the given list of terms.
    def calc_tf(self, tokens):
        tf = defaultdict(int)
        for term, value in Counter(tokens).items():
            if self.normalizer.is_valid_term(term):
                norm_term = self.normalizer.normalize(term)
                tf[norm_term] = tf[norm_term] + value
        return tf

    # Calculate the weighted frequency of the given value.
    def calc_wf(self, value):
        # return 1 + math.log10(value)
        try:
            return self.lookup_wf_calc[value]
        except KeyError:
            v = self.lookup_wf_calc[value] = 1 + math.log10(value)
            return v

    # Calculate the document vector length with respect to the frequencies given.
    @staticmethod
    def calc_vector_length(frequencies):
        return math.sqrt(sum([v ** 2 for v in frequencies.values()]))

    # Get frequency statistics for the individual piece of text.
    def index_text(self, text):
        # Data we have found for the text.
        text_data = {}

        # We may assume that the text has already been cleaned up when we get here.
        # Thus we only have to worry about normalization. I.e, stemming or lemmatization.
        # Before we get ahead of ourselves, start counting the amount of terms, for which we need the terms as an array.
        terms = text.split()

        # Calculate the term frequencies.
        text_data["tf"] = self.calc_tf(terms)
        # noinspection PyArgumentList
        text_data["wf"] = defaultdict(int, {term: self.calc_wf(value) for term, value in text_data["tf"].items()})

        # Add other useful information.
        text_data["number_of_unique_terms"] = len(text_data["tf"])
        text_data["number_of_terms"] = sum(text_data["tf"].values())

        # For now, return the term frequency.
        return text_data

    # Index the query.
    def index_query(self, text, field):
        # Here we do need to normalize the text first...
        cleanup_instance = cleanup.get_cleanup_instance()
        altered_text = cleanup_instance.remove_control_characters(text.lower())
        altered_text = cleanup_instance.remove_punctuation(altered_text)

        # Now we can index it, and update it straight afterwards with the idf data.
        text_data = self.index_text(altered_text)
        self.update_text_result(text_data, self.results["collection"][field]["idf"])
        return text_data

    # Update the paper result to also contain tf.idf and wf.idf plus other useful information.
    def update_text_result(self, text_data, idf):
        # Calculate tf.idf and wf.idf.
        # noinspection PyArgumentList
        text_data["tf.idf"] = defaultdict(int, {term: value * idf[term] for term, value in text_data["tf"].items()})
        # noinspection PyArgumentList
        text_data["wf.idf"] = defaultdict(int, {term: value * idf[term] for term, value in text_data["wf"].items()})

        # Also calculate all the lengths.,
        text_data["vector_lengths"] = {
            "tf": self.calc_vector_length(text_data["tf"]),
            "wf": self.calc_vector_length(text_data["wf"]),
            "tf.idf": self.calc_vector_length(text_data["tf.idf"]),
            "wf.idf": self.calc_vector_length(text_data["wf.idf"]),
        }

    # Index a specific field of the given paper.
    def index_field(self, paper, field):
        # Get the field in question.
        field_value = paper.__getattribute__(field)

        # Index it with the text indexer.
        return self.index_text(field_value)

    # Calculate idf.
    def calc_idf(self, value):
        # return math.log10(len(database.papers) / value)
        try:
            return self.lookup_idf_calc[value]
        except KeyError:
            v = self.lookup_idf_calc[value] = math.log10(len(database.papers) / value)
            return v

    # Index the entirety of the corpus.
    def index_corpus(self, normalizer_name="None", use_stopwords=True):
        # Start a timer for performance measures.
        start = time.time()

        # First set the normalizer.
        self.normalizer = Normalizer(normalizer_name, use_stopwords)

        # Attempt to import the results.
        try:
            self.results = self.load_table_file()
        except FileNotFoundError:
            # The global results, in which we keep the collection and paper results separate.
            self.results = {
                "N": len(database.papers),
                "papers": {},
                "collection": {}
            }

            # We want to index all the papers, which are many fields.
            # Since we don't want to switch our focus all the time, it might be best to loop over the fields first.
            for field in paper_fields:

                # The results found in this field of the paper.
                paper_field_results = {}

                # Now we can take a look at all the papers, and index every single one of them.
                # In the meantime, we keep track of the df and df counters.
                cf = defaultdict(int)
                df = defaultdict(int)
                self.update_status("Indexing field \"" + field + "\"... calculating paper frequencies...")
                for paper in database.papers:
                    # Index the given field, and store the information in the corresponding field.
                    paper_field_results[paper.id] = self.index_field(paper, field)

                    # Update cf and df.
                    for term, value in paper_field_results[paper.id]["tf"].items():
                        cf[term] += value
                        df[term] += 1

                # Now change df to idf.
                self.update_status("Indexing field \"" + field + "\"... calculating idf...")
                idf = defaultdict(int)
                for term, value in df.items():
                    idf[term] = self.calc_idf(value)

                # Add the paper results.
                self.results["papers"][field] = paper_field_results

                # Add the collection results.
                self.results["collection"][field] = {
                    "cf": cf,
                    "df": df,
                    "idf": idf,
                    "vector_lengths": {
                        "cf": self.calc_vector_length(cf),
                        "df": self.calc_vector_length(df),
                        "idf": self.calc_vector_length(idf),
                    },
                    "number_of_unique_terms": len(cf),
                    "number_of_terms": sum(cf.values())
                }

                # Update the paper measures.
                self.update_status("Indexing field \"" + field + "\"... calculating tf.idf and wf.idf measures...")
                for paper_id, value in paper_field_results.items():
                    self.update_text_result(value, idf)

                # Report on the amount of terms found for the field.
                number_of_unique_terms = self.results["collection"][field]["number_of_unique_terms"]
                print("Found", number_of_unique_terms, "unique terms for the field \"" + field + "\".")

            # Store the normalizer table file for later use.
            self.normalizer.create_table_file()

            # Save the file.
            self.create_table_file()

        # Report on how long the indexing took.
        print()
        print("Finished indexing in", time.time() - start, "seconds.")
        print()

    # Dump the results for later use.
    def create_table_file(self):
        self.update_status("Storing index file...")
        with open("../../data/results_" + self.normalizer.normalizer_name + "_" +
                          str(self.normalizer.use_stopwords).lower() + ".pickle", "wb") as output_file:
            pickle.dump(self.results, output_file)

    # Load the results table from disk.
    def load_table_file(self):
        with open("../../data/results_" + self.normalizer.normalizer_name + "_" +
                          str(self.normalizer.use_stopwords).lower() + ".pickle", "rb") as input_file:
            self.update_status("Loading index file...")
            print("Found a file containing the desired index. Importing now.")
            return pickle.load(input_file)

    # Get the normalized values for a specific measurement.
    def get_normalized_paper_values(self, field, measurement):
        normalized_values = {}
        for paper_id, paper_data in self.results["papers"][field].items():
            # Normalize.
            vector_length = paper_data["vector_lengths"][measurement]
            normalized_values[paper_id] = {term: v / vector_length for term, v in paper_data[measurement].items()}
        return normalized_values


if __name__ == "__main__":

    indexer = Indexer()

    for i in range(0, 1):
        indexer.reset()
        start = time.time()
        indexer.index_corpus("Nltk porter stemmer")
        print(time.time() - start)

    # Make sure that the values are correct...
    f = indexer.results
    print(len(f["papers"]["paper_text"][1]["tf"]))
    print(len(f["papers"]["paper_text"][1]["wf"]))
    print(len(f["papers"]["paper_text"][1]["tf.idf"]))
    print(len(f["papers"]["paper_text"][1]["wf.idf"]))
    print(f["papers"]["paper_text"][1]["tf"]["neural"])
    print(f["papers"]["paper_text"][1]["wf"]["neural"])
    print(f["papers"]["paper_text"][1]["tf.idf"]["neural"])
    print(f["papers"]["paper_text"][1]["wf.idf"]["neural"])
    print(f["papers"]["paper_text"][1]["vector_lengths"])
    print(len(f["collection"]["paper_text"]["idf"]))
    print(f["collection"]["paper_text"]["idf"]["neural"])

    for i in range(0, 10):
        start = time.time()
        indexer.get_normalized_paper_values("paper_text", "tf.idf")
        print(time.time() - start)

    # f = indexer.results
    #
    # print(len(f["papers"]["paper_text"][1]["tf"]))
    # print(len(f["papers"]["paper_text"][1]["tf_idf"]))
    # print(f["papers"]["paper_text"][1]["vector_lengths"])
