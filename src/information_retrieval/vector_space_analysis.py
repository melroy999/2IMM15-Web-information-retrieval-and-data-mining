# field: paper_tfs, idf_collection, idf_collection_length

# The scoring measures the user can choose from.
from collections import defaultdict

import math

from import_data import database

scoring_measures = {"tf": 0, "wf": 1, "tf.idf": 2, "wf.idf": 3}


def cosine_similarity(target_collection, target_collection_length, candidate_collection,
                      candidate_collection_length):
    # Find which of the two has the least terms, and use that collection as iteration base.
    if len(candidate_collection) > len(target_collection):
        iteration_collection = target_collection.keys()
    else:
        iteration_collection = candidate_collection.keys()

    # Iterate over all terms in the collection, and calculate the score.
    dot_product = 0
    for term in iteration_collection:
        dot_product += target_collection[term] * candidate_collection[term]

    # Normalize the result by using the length of the document vectors.
    normalization_factor = target_collection_length * candidate_collection_length

    # Report the score.
    return dot_product / normalization_factor


def cosine_paper_similarity(indexer_results, field, list_id_target, list_id_candidate, scoring_measure_id):
    # First get the tf or wf values.
    tf_wf_collection = indexer_results[field][0]
    target_collection = tf_wf_collection[list_id_target][scoring_measure_id % 4]
    target_collection_length = tf_wf_collection[list_id_target][4 + scoring_measure_id % 4]
    candidate_collection = tf_wf_collection[list_id_candidate][scoring_measure_id % 4]
    candidate_collection_length = tf_wf_collection[list_id_candidate][4 + scoring_measure_id % 4]

    # Report the score.
    return cosine_similarity(target_collection, target_collection_length, candidate_collection,
                             candidate_collection_length)


def cosine_query_similarity(indexer_results, field, query_collection, query_collection_length,
                            list_id_candidate, scoring_measure_id):
    # First get the tf or wf values.
    tf_wf_collection = indexer_results[field][0]
    candidate_collection = tf_wf_collection[list_id_candidate][scoring_measure_id % 4]
    candidate_collection_length = tf_wf_collection[list_id_candidate][4 + scoring_measure_id % 4]

    return cosine_similarity(query_collection, query_collection_length, candidate_collection,
                             candidate_collection_length)


class EmptyQueryException(Exception):
    pass


def query_papers_search(query, indexer, field, scoring_measure_id):
    # First normalize and tokenize the query.
    query_frequency_data = indexer.index_query(query)

    # The query can end up empty because of tokenization. So throw an exception of this is the case.
    if len(query_frequency_data[0]) == 0:
        raise EmptyQueryException()

    # We first have to calculate the tf.idf and wf.idf components.
    query_frequency_data = indexer.calculate_tf_idf(query_frequency_data, indexer.results[field][1])

    # Get the tf or wf value, depending on the comparison mode.
    query_collection = query_frequency_data[scoring_measure_id % 4]
    query_collection_length = query_frequency_data[4 + scoring_measure_id % 4]

    # Iterate over all papers, and gather the scores.
    scores = {}
    for i, paper in enumerate(database.papers):
        score = cosine_query_similarity(indexer.results, field, query_collection, query_collection_length,
                                        i, scoring_measure_id)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def similar_papers_search(query, indexer, field, scoring_measure_id):
    try:
        # Check if the query is an int.
        paper_list_id = database.paper_id_to_list_id[int(query)]
    except ValueError:
        # Do a query search and find the most appropriate paper.
        target_paper_scores = query_papers_search(query, indexer, "title", scoring_measure_id)

        # Select the best paper. It may not exist.
        try:
            paper_list_id = database.paper_id_to_list_id[target_paper_scores[0][0]]
        except IndexError:
            print("Paper not found. Please try again!")
            return

    # Report which paper we have found.
    paper = database.papers[paper_list_id]
    print("Target paper: #" + str(paper.id) + " \"" + paper.title + "\"")

    # Iterate over all papers, and gather the scores.
    scores = {}
    for i, paper in enumerate(database.papers):
        score = cosine_paper_similarity(indexer.results, field, paper_list_id, i, scoring_measure_id)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search(query, indexer, field, scoring_measure="tf", similar_document_search=False):
    # Get the measure id.
    scoring_measure_id = scoring_measures[scoring_measure]

    if similar_document_search:
        # Search for similar papers. First find the target paper.
        return similar_papers_search(query, indexer, field, scoring_measure_id)
    else:
        # Search for query.
        return query_papers_search(query, indexer, field, scoring_measure_id)
