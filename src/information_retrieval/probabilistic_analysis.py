import string
from collections import defaultdict
from functools import reduce
from operator import mul

import math

import time

from cleanup_module import cleanup
from information_retrieval.indexer import Indexer
from information_retrieval.vector_space_analysis import EmptyQueryException


# A function we will use when we want all documents to be equally probable.
def document_probability_equal_weight(d_l, c_l):
    return 1


# A function we will use when we want longer documents to be more probable.
def document_probability_length_weight(d_l, c_l):
    return d_l / c_l


# The different modes we have for calculating P(d).
document_probability_modes = {
    "Equal probability": document_probability_equal_weight,
    "Document length based": document_probability_length_weight
}

# The available search models.
search_modes = ["Mixture model", "Okapi BM25", "Okapi BM25+"]

# The available modes for the IDF values in okapi.
okapi_idf_modes = ["Pure IDF", "Okapi IDF", "Okapi floored IDF", "Summand 0 flooring"]


class ProbabilisticAnalysis:
    # Initialize all the class variables.
    def __init__(self):
        self._lambda = 0
        self.k_1 = 1
        self.b = 1
        self.epsilon = 0.5
        self.delta = 0
        self.okapi_summand = None
        self.okapi_idf_mode = None
        self.idf = defaultdict(int)

    # Calculate the probability for a given document, using smooth mixed multinomial.
    # Here low lambda is more suitable for longer queries, and high lambda is more suitable for queries that desire all
    # query terms to be present.
    def calc_mixture_model_probability(self, tokens, paper_frequencies, collection_frequencies, p_d):
        # We will only need the term frequencies for the specific term and our total number of tokens.
        tf = paper_frequencies["tf"]
        dl = paper_frequencies["number_of_terms"]

        cf = collection_frequencies["cf"]
        cl = collection_frequencies["number_of_terms"]

        # Calculate the probability for this paper. here p_d is the probability that this document is chosen.
        # The latter is the probability that the query would occur according to the document model.
        return p_d(dl, cl) * reduce(mul, [self.smooth_mixed_multinomial(term, tf, dl, cf, cl) for term in tokens])

    # A mixture between document likelihood and collection likelihood, using a lambda as the ratio between the two.
    def smooth_mixed_multinomial(self, term, tf, dl, cf, cl):
        return self._lambda * (tf[term] / dl) + (1 - self._lambda) * (cf[term] / cl)

    # Do the search for the given query, using the unigram language mixture model.
    def search_mixture_model(self, query_tokens, indexer, field, document_probability_mode_name):
        # We will need the same collection frequencies for all calculations.
        collection_frequencies = indexer.results["collection"][field]

        # Set the document probability mode.
        p_d = document_probability_modes[document_probability_mode_name]

        # Check all papers.
        chances = {}
        for paper_id, paper_frequencies in indexer.results["papers"][field].items():
            # Calculate the probability.
            chances[paper_id] = self.calc_mixture_model_probability(query_tokens, paper_frequencies,
                                                                    collection_frequencies, p_d)

        # Now we can find the papers with the highest chance.
        return sorted(chances.items(), key=lambda x: x[1], reverse=True)

    # Here n is the number of documents, and avg_dl is the average document length.
    def calc_okapi_bm25(self, tokens, paper_frequencies, collection_frequencies, n, avg_dl, k_1, b):
        # We will only need the term frequencies and the document length.
        tf = paper_frequencies["tf"]
        dl = paper_frequencies["number_of_terms"]

        # We will also need the document frequencies.
        df = collection_frequencies["df"]

        # We have to observe this for all the tokens, and take the sum of all the scores.
        return sum([self.okapi_summand(term, tf[term], dl, df[term], n, avg_dl, k_1, b) for term in tokens])

    # Calculate the summand that occurs within the okapi ranking function.
    def calc_okapi_summand(self, term, tf_t, dl, df_t, n, avg_dl, k_1, b):
        return self.okapi_idf_mode(n, df_t, term) * self.calc_something(tf_t, dl, avg_dl, k_1, b)

    # Calculate the summand that occurs within the okapi ranking function, which floors to 0.
    def calc_okapi_summand_floored(self, term, tf_t, dl, df_t, n, avg_dl, k_1, b):
        return max(self.okapi_idf_mode(n, df_t, term) * self.calc_something(tf_t, dl, avg_dl, k_1, b), 0.0)

    # Instead of using the idf we already calculated, we will use the formula proposed in the slides.
    @staticmethod
    def calc_okapi_idf(n, df_t, term):
        return math.log10((n - df_t + 0.5) / (df_t + 0.5))

    # The same as okapi idf, but now with a floor value.
    def calc_okapi_idf_with_floor(self, n, df_t, term):
        return max(math.log10((n - df_t + 0.5) / (df_t + 0.5)), self.epsilon)

    def fetch_idf(self, n, df_t, term):
        return self.idf[term]

    # Not sure how to name the second section of the okapi formula... we calculate it here.
    def calc_something(self, tf_t, dl, avg_dl, k_1, b):
        dividend = tf_t * (k_1 + 1)
        divisor = tf_t + k_1 * (1 - b + b * dl / avg_dl)
        return (dividend / divisor) + self.delta

    # Do the search for the given query, using the okapi bm25+ model.
    def search_okapi_bm25(self, query_tokens, indexer, field, k_1, b):
        # We will need the same collection frequencies for all calculations.
        collection_frequencies = indexer.results["collection"][field]

        # Calculate the values we might need.
        n = len(indexer.results["papers"][field])
        avg_dl = sum([value["number_of_terms"] for value in indexer.results["papers"][field].values()]) / n

        # Check all papers.
        chances = {}
        for paper_id, paper_frequencies in indexer.results["papers"][field].items():
            # Calculate the probability.
            chances[paper_id] = \
                self.calc_okapi_bm25(query_tokens, paper_frequencies, collection_frequencies, n, avg_dl, k_1, b)

        # Now we can find the papers with the highest chance.
        return sorted(chances.items(), key=lambda x: float(x[1]), reverse=True)

    # Do the search for the given query, using the unigram language model.
    def search(self, query, indexer, field,
               search_mode_name=search_modes[0],
               document_probability_mode_name="Equal Probability",
               okapi_idf_mode_name=okapi_idf_modes[1],
               remove_duplicates=True,
               _lambda=0.0,
               k_1=2.0, b=0.75, delta=1.0, epsilon=0.5):
        # Now we can split the query, and we will have our tokens.
        query_tokens = indexer.normalize_and_tokenize_query(query)

        # Remove duplicates if required.
        if remove_duplicates:
            query_tokens = set(query_tokens)

        # The query can end up empty because of tokenization. So throw an exception of this is the case.
        if len(query_tokens) == 0:
            raise EmptyQueryException()

        # Set all of the passed parameters in the class.
        self.k_1 = k_1
        self.b = b
        self._lambda = _lambda
        self.okapi_summand = self.calc_okapi_summand
        self.okapi_idf_mode = self.calc_okapi_idf
        self.epsilon = epsilon
        self.idf = defaultdict(int)

        # Values we want to reset to default on every call.
        self.delta = 0

        # Options: ["Pure IDF", "Okapi IDF", "Okapi floored IDF", "Summand 0 flooring"]
        # Now we should decide which of the idf modes we want to use.
        if okapi_idf_mode_name == "Summand 0 flooring":
            # We want to use the default idf mode, but we have to change the summand.
            self.okapi_summand = self.calc_okapi_summand_floored
        elif okapi_idf_mode_name == "Pure IDF":
            # Change the okapi_idf_mode to the correct one, and set the idf table.
            self.okapi_idf_mode = self.fetch_idf
            self.idf = indexer.results["collection"][field]["idf"]
        elif okapi_idf_mode_name == "Okapi floored IDF":
            # Change the okapi_idf_mode to the correct one.
            self.okapi_idf_mode = self.calc_okapi_idf_with_floor
        else:
            # Otherwise, change nothing, we can use the default configuration.
            pass

        # Let the appropriate option calculate the probability.
        if search_mode_name == "Mixture model":
            print("Executing \"mixed probability model\" with the following parameters:")
            print("- Target field:", field)
            print("- Remove duplicate terms in query:", remove_duplicates)
            print("- Document probability mode:", document_probability_mode_name)
            print(u"- \u03BB:", _lambda)
            print()
            return self.search_mixture_model(query_tokens, indexer, field, document_probability_mode_name)
        elif search_mode_name == "Okapi BM25":
            self.print_okapi_mode_parameters(search_mode_name, okapi_idf_mode_name, remove_duplicates, field)
            return self.search_okapi_bm25(query_tokens, indexer, field, k_1, b)
        elif search_mode_name == "Okapi BM25+":
            self.print_okapi_mode_parameters(search_mode_name, okapi_idf_mode_name, remove_duplicates, field)
            # Here we need to set delta to the user defined value.
            self.delta = delta
            return self.search_okapi_bm25(query_tokens, indexer, field, k_1, b)
        else:
            print("Search mode unknown.")

    def print_okapi_mode_parameters(self, search_mode_name, okapi_idf_mode_name, remove_duplicates, field):
        print("Executing \"" + search_mode_name + "\" with the following parameters:")
        print("- Target field:", field)
        print("- Remove duplicate terms in query:", remove_duplicates)
        print(u"- k\u2081:", self.k_1)
        print(u"- b:", self.b)
        if search_mode_name == "Okapi BM25+":
            print(u"- \u03B4:", self.delta)
        if okapi_idf_mode_name == "Summand 0 flooring":
            print("- Summand mode: 0 flooring")
        else:
            print("- IDF mode:", okapi_idf_mode_name)
            if okapi_idf_mode_name == "Okapi floored IDF":
                print(u"- \u03B5:", self.epsilon)
        print()


if __name__ == "__main__":
    _indexer = Indexer()
    _indexer.index_corpus("None", True)
    start = time.time()

    # Create a probabilistic analysis instance.
    an = ProbabilisticAnalysis()

    # Search terms we want to use.
    search_terms = ["Cheese", "Neural", "Bitch"]

    # We kinda want to try all options...
    for search_term in search_terms:
        for mode in search_modes:
            if mode == "Mixture model":
                # We have the document probability modes to check.
                for probability_mode in document_probability_modes:
                    print(search_term, mode, probability_mode, sep=', ')
                    result = an.search(search_term, _indexer, "paper_text", search_mode_name=mode,
                                       document_probability_mode_name=probability_mode)
                    print(result[:10])
                    print(result[-10:])
            else:
                # We have multiple idf modes to check.
                for idf_mode in okapi_idf_modes:
                    result = an.search(search_term, _indexer, "paper_text", search_mode_name=mode,
                                       okapi_idf_mode_name=idf_mode)
                    print(search_term, mode, idf_mode, sep=', ')
                    print(result[:10])
                    print(result[-10:])
        print()
