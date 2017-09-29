import math


def cosine_similarity_tf(collection_frequency_data, term_frequency_data, query_frequency_data):
    # The terms in the query.
    query_terms = set(query_frequency_data)

    # Calculate the score.
    # Remember that the tf_d * tf_q of a term not in the document is 0, so we essentially only have to look at the
    # terms in the query.
    scores = {
        paper_id: (sum(paper_term_frequencies[term][3] * query_frequency_data[term][3] for term in query_terms), {term: paper_term_frequencies[term][3] * query_frequency_data[term][3] for term in query_terms})
        for paper_id, paper_term_frequencies in term_frequency_data.items()
    }

    # Remove 0 scores.
    scores = {paper_id: (value, mapping) for paper_id, (value, mapping) in scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}

    # Sort the scores.
    return sorted(scores.items(), key=lambda x: x[1][0], reverse=True)


def cosine_similarity_tf_idf(collection_frequency_data, term_frequency_data, query_frequency_data):
    # The terms in the query.
    query_terms = set(query_frequency_data)

    # Create a dictionary to store the scores in.
    scores = {}

    # Calculate the score.
    for paper_id, paper_term_frequencies in term_frequency_data.items():
        # We have to calculate the if.idf value and normalize it.
        # These calculations should be scarce depending on the size of the query.
        # Take neither of the original values normalized.
        tf_idf = {term: paper_term_frequencies[term][1] * collection_frequency_data[term][2] for term in query_terms}

        # Now calculate the length.
        tf_idf_length = calc_vector_length(tf_idf.values())

        # We might divide by 0, if so just give the score 0.
        try:
            # Now normalize the if.idf array.
            n_tf_idf = {term: value / tf_idf_length for term, value in tf_idf.items()}

            # Now multiply the papers if_idf by the tf of the query (dot product).
            scores[paper_id] = (sum(n_tf_idf[term] * query_frequency_data[term][3] for term in query_terms), {term: n_tf_idf[term] * query_frequency_data[term][3] for term in query_terms})
        except ZeroDivisionError:
            scores[paper_id] = (0.0, None)

    # Remove 0 scores.
    scores = {paper_id: (value, mapping) for paper_id, (value, mapping) in scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}

    # Sort the scores.
    return sorted(scores.items(), key=lambda x: x[1][0], reverse=True)


# Calculate the vector length of an term frequency table.
def calc_vector_length(values):
    return math.sqrt(sum([x*x for x in values]))
