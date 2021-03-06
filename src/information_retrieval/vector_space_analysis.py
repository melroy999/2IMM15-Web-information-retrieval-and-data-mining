import math
from import_data import database

# The scoring measures the user can choose from.
scoring_measures = ["tf", "wf", "tf.idf", "wf.idf"]


# Calculate the inner product of the two vectors.
def inner_product(target_frequencies, target_vector_length, candidate_frequencies, candidate_vector_length):
    # Find the common terms.
    common_terms = set(target_frequencies).intersection(candidate_frequencies)

    # Now calculate the dot product.
    dot_product = 0
    for term in common_terms:
        dot_product += target_frequencies[term] * candidate_frequencies[term]
    return dot_product


# Calculate the dice coefficient of the two vectors.
def dice_coefficient(target_frequencies, target_vector_length, candidate_frequencies, candidate_vector_length):
    # We need the dot product.
    dot_product = inner_product(target_frequencies, 0, candidate_frequencies, 0)

    # We have to divide double the dot product by the square of the two document lengths.
    try:
        return 2 * dot_product / (target_vector_length**2 + candidate_vector_length**2)
    except ZeroDivisionError:
        return 0


# Calculate the cosine coefficient of the two vectors.
def cosine_coefficient(target_frequencies, target_vector_length, candidate_frequencies, candidate_vector_length):
    # We need the dot product.
    dot_product = inner_product(target_frequencies, 0, candidate_frequencies, 0)

    # We have to divide the dot product by the product of the two document vector lengths.
    try:
        return dot_product / (target_vector_length * candidate_vector_length)
    except ZeroDivisionError:
        return 0


# Calculate the Jaccard coefficient.
def jaccard_coefficient(target_frequencies, target_vector_length, candidate_frequencies, candidate_vector_length):
    # We need the dot product.
    dot_product = inner_product(target_frequencies, 0, candidate_frequencies, 0)

    # We have to divide the dot product by the square of the two document vector lengths, minus the dot product.
    try:
        return dot_product / (target_vector_length**2 + candidate_vector_length**2 - dot_product)
    except ZeroDivisionError:
        return 0


# The similarity measures we have.
similarity_measures = {
    "Inner product": inner_product,
    "Dice coefficient": dice_coefficient,
    "Cosine coefficient": cosine_coefficient,
    "Jaccard coefficient": jaccard_coefficient
}


# Measure the similarity between two papers.
def paper_similarity(data, field, target_paper_id, candidate_paper_id, scoring_measure, similarity_measure):
    # First get the tf or wf values.
    papers_data = data["papers"][field]
    target_frequencies = papers_data[target_paper_id][scoring_measure]
    target_vector_length = papers_data[target_paper_id]["vector_lengths"][scoring_measure]
    candidate_frequencies = papers_data[candidate_paper_id][scoring_measure]
    candidate_vector_length = papers_data[candidate_paper_id]["vector_lengths"][scoring_measure]

    # Report the score.
    return similarity_measure(target_frequencies, target_vector_length, candidate_frequencies, candidate_vector_length)


# Measure the similarity between the query and a paper.
def query_similarity(data, field, query_collection, query_collection_length,
                     candidate_paper_id, scoring_measure, similarity_measure):
    # First get the tf or wf values.
    paper_frequency_data = data["papers"][field][candidate_paper_id]
    candidate_frequencies = paper_frequency_data[scoring_measure]
    candidate_vector_length = paper_frequency_data["vector_lengths"][scoring_measure]

    return similarity_measure(query_collection, query_collection_length, candidate_frequencies, candidate_vector_length)


# An exception we can throw when the query is empty.
class EmptyQueryException(Exception):
    pass


# Search papers that are similar to the query.
def query_papers_search(query, indexer, field, scoring_measure, similarity_measure):
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
        score = query_similarity(indexer.results, field, query_frequencies,
                                 query_vector_length, paper.id, scoring_measure, similarity_measure)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Search papers that are similar to the target paper.
def similar_papers_search(query, indexer, field, scoring_measure, similarity_measure):
    try:
        # Check if the query is an int.
        paper_id = int(query)
    except ValueError:
        # Do a query search and find the most appropriate paper.
        target_paper_scores = query_papers_search(query, indexer, "title", scoring_measure, similarity_measure)

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
        score = paper_similarity(indexer.results, field, paper_id, paper.id, scoring_measure, similarity_measure)

        # Filter out scores that are close to zero.
        if not math.isclose(score, 0.0, abs_tol=1e-19):
            scores[paper.id] = score

    # sort the scores and return.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Execute a vector space search.
def search(query, indexer, field, scoring_measure_name=scoring_measures[0], similar_document_search=False,
           similarity_measure_name="Cosine coefficient"):
    if similar_document_search:
        # Search for similar papers. First find the target paper.
        return similar_papers_search(query, indexer, field, scoring_measure_name,
                                     similarity_measures[similarity_measure_name])
    else:
        # Search for query.
        return query_papers_search(query, indexer, field, scoring_measure_name,
                                   similarity_measures[similarity_measure_name])
