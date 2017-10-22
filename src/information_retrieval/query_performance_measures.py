import time

import sys

import os

import gc

from information_retrieval.indexer import Indexer
import information_retrieval.boolean_analysis as boolean_retrieval
import information_retrieval.vector_space_analysis as vector_retrieval
import information_retrieval.probabilistic_analysis as probability_retrieval


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def print_boolean_space_array_data(data, query, result_count):
    enablePrint()
    print("\\multicolumn{4}{l}{" + query.replace("_", "\\_") + "}\\\\")
    print(result_count, "%.8f" % min(data), "%.8f" % (sum(data) / len(data)), "%.8f" % max(data), sep=" & ", end="\\\\\n")
    blockPrint()


def print_vector_space_array_data(data, query, measurement, similarity, result_count):
    enablePrint()
    print(measurement, similarity, "%.8f" % min(data), "%.8f" % (sum(data) / len(data)), "%.8f" % max(data),
          sep=" & ", end="\\\\\n")
    blockPrint()


def print_probability_space_array_data(data, query, result_count, search_method, modifier):
    enablePrint()
    print(search_method, modifier, result_count, "%.8f" % min(data), "%.8f" % (sum(data) / len(data)),
          "%.8f" % max(data), sep=" & ", end="\\\\\n")
    blockPrint()


# We use only one indexer.
indexer = Indexer()
indexer.index_corpus("Nltk porter stemmer")
blockPrint()


def measurement_wrapper(function, timings):
    start = time.time()
    matches = function()
    timings.append(time.time() - start)
    return matches


def boolean_search_test(query):
    timings = []
    result = None
    for i in range(0, 1000):
        result = measurement_wrapper(lambda: boolean_retrieval.search(query, indexer, "paper_text"), timings)
    print_boolean_space_array_data(timings, query, len(result))


def vector_space_search_test(query, measurement, similarity):
    timings = []
    result = None
    for i in range(0, 100):
        result = measurement_wrapper(lambda: vector_retrieval.search(query, indexer, "paper_text",
                                                                     scoring_measure_name=measurement,
                                                                     similarity_measure_name=similarity,
                                                                     similar_document_search=True), timings)
    print_vector_space_array_data(timings, query, measurement, similarity, len([r for r in result if r > 0]))


def probability_space_search_test(query, search_method, modifier):
    timings = []
    result = None
    for i in range(0, 1000):
        result = measurement_wrapper(lambda: probability_retrieval.ProbabilisticAnalysis().
                                     search(query, indexer, "paper_text",
                                            search_mode_name=search_method,
                                            document_probability_mode_name=modifier,
                                            okapi_idf_mode_name=modifier, _lambda=0.5), timings)
    print_probability_space_array_data(timings, query, len(result), search_method, modifier)


# We have multiple probability measures. Iterate over them.
for search_method in probability_retrieval.search_modes:
    if search_method == "Mixture model":
        modifiers = probability_retrieval.document_probability_modes.keys()
    else:
        modifiers = probability_retrieval.okapi_idf_modes

    for modifier in modifiers:
        probability_space_search_test("Neural networks in neuroscience and deep learning", search_method, modifier)
