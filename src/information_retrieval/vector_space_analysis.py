import math
from import_data import database

# The scoring measures the user can choose from.
scoring_measures = ["tf", "wf", "tf.idf", "wf.idf"]


def cosine_similarity(target_frequencies, target_vector_length, candidate_frequencies,
                      candidate_vector_length):
    # Find which of the two has the least terms, and use that collection as iteration base.
    if len(candidate_frequencies) > len(target_frequencies):
        iteration_collection = target_frequencies.keys()
    else:
        iteration_collection = candidate_frequencies.keys()

    # Iterate over all terms in the collection, and calculate the score.
    dot_product = 0
    for term in iteration_collection:
        dot_product += target_frequencies[term] * candidate_frequencies[term]

    # If the dot product is 0, we should just return 0.
    if dot_product == 0:
        return 0

    # Normalize the result by using the length of the document vectors.
    normalization_factor = target_vector_length * candidate_vector_length

    # Report the score.
    return dot_product / normalization_factor


def cosine_paper_similarity(data, field, target_paper_id, candidate_paper_id, scoring_measure):
    # First get the tf or wf values.
    papers_data = data["papers"][field]
    target_frequencies = papers_data[target_paper_id][scoring_measure]
    target_vector_length = papers_data[target_paper_id]["vector_lengths"][scoring_measure]
    candidate_frequencies = papers_data[candidate_paper_id][scoring_measure]
    candidate_vector_length = papers_data[candidate_paper_id]["vector_lengths"][scoring_measure]

    # Report the score.
    return cosine_similarity(target_frequencies, target_vector_length, candidate_frequencies,
                             candidate_vector_length)


def cosine_query_similarity(data, field, query_collection, query_collection_length,
                            candidate_paper_id, scoring_measure):
    # First get the tf or wf values.
    paper_frequency_data = data["papers"][field][candidate_paper_id]
    candidate_frequencies = paper_frequency_data[scoring_measure]
    candidate_vector_length = paper_frequency_data["vector_lengths"][scoring_measure]

    return cosine_similarity(query_collection, query_collection_length, candidate_frequencies,
                             candidate_vector_length)


class EmptyQueryException(Exception):
    pass


def query_papers_search(query, indexer, field, scoring_measure):
    # First normalize and tokenize the query.
    query_frequency_data = indexer.index_query(query, field)

    # The query can end up empty because of tokenization. So throw an exception of this is the case.
    if len(query_frequency_data["tf"]) == 0:
        raise EmptyQueryException()

    # Get the tf or wf value, depending on the comparison mode.
    query_frequencies = query_frequency_data[scoring_measure]
    query_vector_length = query_frequency_data["vector_lengths"][scoring_measure]

    # Iterate over all papers, and gather the scores.
    scores = {}
    for paper in database.papers:
        score = cosine_query_similarity(indexer.results, field, query_frequencies,
                                        query_vector_length, paper.id, scoring_measure)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def similar_papers_search(query, indexer, field, scoring_measure):
    try:
        # Check if the query is an int.
        paper_id = int(query)
    except ValueError:
        # Do a query search and find the most appropriate paper.
        target_paper_scores = query_papers_search(query, indexer, "title", scoring_measure)

        # Select the best paper. It may not exist.
        try:
            paper_id = target_paper_scores[0][0]
        except IndexError:
            print("Paper not found. Please try again!")
            return

    # Report which paper we have found.
    paper = database.paper_id_to_paper[paper_id]
    print("Target paper: #" + str(paper.id) + " \"" + paper.stored_title + "\"")

    # Iterate over all papers, and gather the scores.
    scores = {}
    for paper in database.papers:
        score = cosine_paper_similarity(indexer.results, field, paper_id, paper.id, scoring_measure)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search(query, indexer, field, scoring_measure=scoring_measures[0], similar_document_search=False):
    if similar_document_search:
        # Search for similar papers. First find the target paper.
        return similar_papers_search(query, indexer, field, scoring_measure)
    else:
        # Search for query.
        return query_papers_search(query, indexer, field, scoring_measure)