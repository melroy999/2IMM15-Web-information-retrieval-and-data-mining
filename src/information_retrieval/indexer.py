import gc
from collections import Counter, defaultdict

import time

from import_data import database
import cleanup_module.cleanup as cleanup

# The fields we target in the papers.
from information_retrieval.normalizer import Normalizer

# The fields we may target during the indexing.
paper_fields = ["title", "abstract", "paper_text"]


class Indexer(object):
    # Holds the results of the indexing.
    results = None

    # Holds the normalizer last used in this indexer.
    normalizer = None

    # Initializes the indexer.
    def __init__(self):
        # Load the papers and authors using the cleanup module, as we don't want all the gibberish.
        cleanup.get_cleanup_instance(database).clean(database.papers)
        self.lookup_wf_calc = {}
        self.lookup_idf_calc = {}

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
        return math.sqrt(sum([v**2 for v in frequencies.values()]))

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
        text_data["wf"] = {term: self.calc_wf(value) for term, value in text_data["tf"].items()}

        # Add other useful information.
        text_data["number_of_unique_terms"] = len(text_data["tf"])
        text_data["number_of_terms"] = sum(text_data["tf"].values())

        # For now, return the term frequency.
        return text_data

    # Update the paper result to also contain tf.idf and wf.idf plus other useful information.
    def update_text_result(self, text_data, idf):
        # Calculate tf.idf and wf.idf.
        text_data["tf.idf"] = {term: value * idf[term] for term, value in text_data["tf"].items()}
        text_data["wf.idf"] = {term: value * idf[term] for term, value in text_data["wf"].items()}

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
        # First set the normalizer.
        self.normalizer = Normalizer(normalizer_name, use_stopwords)

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
            for paper in database.papers:
                # Index the given field, and store the information in the corresponding field.
                paper_field_results[paper.id] = self.index_field(paper, field)

                # Update cf and df.
                for term, value in paper_field_results[paper.id]["tf"].items():
                    cf[term] += value
                    df[term] += 1

            # Now change df to idf.
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
            for paper_id, value in paper_field_results.items():
                self.update_text_result(value, idf)

        # Store the normalizer table file for later use.
        self.normalizer.create_table_file()


import math

if __name__ == "__main__":

    indexer = Indexer()

    for i in range(0, 10):
        indexer.reset()
        start = time.time()
        indexer.index_corpus("Nltk porter stemmer")
        print(time.time() - start)

    # Make sure that the values are correct...
    f = indexer.results
    print(len(f["papers"]["paper_text"][1]["tf"]))
    print(len(f["papers"]["paper_text"][1]["tf.idf"]))
    print(f["papers"]["paper_text"][1]["vector_lengths"])

    # f = indexer.results
    #
    # print(len(f["papers"]["paper_text"][1]["tf"]))
    # print(len(f["papers"]["paper_text"][1]["tf_idf"]))
    # print(f["papers"]["paper_text"][1]["vector_lengths"])
