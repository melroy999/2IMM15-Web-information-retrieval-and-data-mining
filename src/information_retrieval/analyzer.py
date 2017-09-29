import math


def cosine_similarity(collection_frequency_data, term_frequency_data, query_frequency_data):
    # The terms in the query.
    query_terms = set(query_frequency_data)

    # Calculate the score.
    scores = {
        paper_id: sum(term_frequencies[term][2] * query_frequency_data[term][2] for term in query_terms)
        for paper_id, term_frequencies in term_frequency_data.items()
    }

    # Remove 0 scores.
    scores = {term: value for term, value in scores.items() if not math.isclose(value, 0.0, abs_tol=1e-09)}

    # Sort the scores.
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Calculate the vector length of an term frequency table.
def term_frequency_table_vector_length(frequency_table):
    return math.sqrt(sum([x*x for x in frequency_table.values()]))
