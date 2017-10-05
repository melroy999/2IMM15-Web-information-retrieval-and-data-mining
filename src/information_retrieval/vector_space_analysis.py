# field: paper_tfs, idf_collection, idf_collection_length

# The scoring measures the user can choose from.
from collections import defaultdict
from import_data import database

scoring_measures = {"tf": 0, "wf": 1, "tf.idf": 2, "wf.idf": 3}


def cosine_similarity(target_collection, target_collection_length, candidate_collection,
                      candidate_collection_length, idf_collection, idf_collection_length):
    # Find which of the two has the least terms, and use that collection as iteration base.
    if len(candidate_collection) > len(target_collection):
        iteration_collection = target_collection.keys()
    else:
        iteration_collection = candidate_collection.keys()

    # Iterate over all terms in the collection, and calculate the score.
    dot_product = 0
    for term in iteration_collection:
        a = target_collection[term]
        b = candidate_collection[term]
        c = idf_collection[term]
        d = idf_collection[term][2]
        dot_product += target_collection[term] * candidate_collection[term] * idf_collection[term][2] ** 2

    # Normalize the result by using the length of the document vectors.
    normalization_factor = target_collection_length * candidate_collection_length * idf_collection_length ** 2

    # Report the score.
    return dot_product / normalization_factor


def cosine_paper_similarity(indexer_results, field, list_id_target, list_id_candidate, scoring_measure_id):
    # First get the tf or wf values.
    tf_wf_collection = indexer_results[field][0]
    target_collection = tf_wf_collection[list_id_target][scoring_measure_id % 2]
    target_collection_length = tf_wf_collection[list_id_target][2 + scoring_measure_id % 2]
    candidate_collection = tf_wf_collection[list_id_candidate][scoring_measure_id % 2]
    candidate_collection_length = tf_wf_collection[list_id_candidate][2 + scoring_measure_id % 2]

    # Get the idf table, which is empty when we have scoring measures tf or wf.
    # In the case that it is empty, it should always return 1.
    idf_collection, idf_collection_length = choose_idf_collection(field, indexer_results, scoring_measure_id)

    # Report the score.
    return cosine_similarity(target_collection, target_collection_length, candidate_collection,
                             candidate_collection_length, idf_collection, idf_collection_length)


def cosine_query_similarity(indexer_results, field, query_collection, query_collection_length,
                            list_id_candidate, scoring_measure_id):
    # First get the tf or wf values.
    tf_wf_collection = indexer_results[field][0]
    candidate_collection = tf_wf_collection[list_id_candidate][scoring_measure_id % 2]
    candidate_collection_length = tf_wf_collection[list_id_candidate][2 + scoring_measure_id % 2]

    # Get the idf table, which is empty when we have scoring measures tf or wf.
    # In the case that it is empty, it should always return 1.
    idf_collection, idf_collection_length = choose_idf_collection(field, indexer_results, scoring_measure_id)

    return cosine_similarity(query_collection, query_collection_length, candidate_collection,
                             candidate_collection_length, idf_collection, idf_collection_length)


def choose_idf_collection(field, indexer_results, scoring_measure_id):
    if scoring_measure_id > 1:
        idf_collection = indexer_results[field][1]
        idf_collection_length = indexer_results[field][2]
    else:
        idf_collection = defaultdict(lambda: (0, 0, 1))
        idf_collection_length = 1
    return idf_collection, idf_collection_length


def query_papers_search(query, indexer, field, scoring_measure_id):
    # First normalize and tokenize the query.
    query_frequency_data = indexer.index_query(query)

    # Get the tf or wf value, depending on the comparison mode.
    query_collection = query_frequency_data[scoring_measure_id % 2]
    query_collection_length = query_frequency_data[2 + scoring_measure_id % 2]

    # Iterate over all papers, and gather the scores.
    scores = {}
    for i, paper in enumerate(database.papers):
        scores[paper.id] = cosine_query_similarity(indexer.results, field, query_collection, query_collection_length,
                                                   i, scoring_measure_id)

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
        except KeyError:
            print("Paper not found. Please try again!")
            return

    # Iterate over all papers, and gather the scores.
    scores = {}
    for i, paper in enumerate(database.papers):
        scores[paper.id] = cosine_paper_similarity(indexer.results, field, paper_list_id, i, scoring_measure_id)

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
