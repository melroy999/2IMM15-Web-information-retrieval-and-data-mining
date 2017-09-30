import math

import information_retrieval.indexer as indexer
from import_data import database


########################################################################################################################
# Generic functions used during all similarity measures.
########################################################################################################################


# Calculate the vector length of a term frequency table.
def calc_vector_length(terms, term_to_score):
    return math.sqrt(sum([term_to_score[term] ** 2 for term in terms]))


# Calculate the dot product of the two documents.
def calc_dot_product(terms, term_to_score_doc_1, term_to_score_doc_2):
    return sum(term_to_score_doc_1[term] * term_to_score_doc_2[term] for term in terms)


# Calculate the cosine similarity of two data sets.
def cosine_similarity(term_to_score_doc_1, term_to_score_doc_2, intersection=True):
    # We first need to find the complete set of terms.
    # It holds for all measures that missing terms get the score zero, thus we can take the intersection of sets.
    # Just to be sure we made this toggle-able.
    if intersection:
        terms = set(term_to_score_doc_1).intersection(term_to_score_doc_2)
    else:
        terms = set(term_to_score_doc_1).union(term_to_score_doc_2)

    # Calculate the dot product and vector lengths of the two data sets.
    dot_product = calc_dot_product(terms, term_to_score_doc_1, term_to_score_doc_2)

    # We assume that the input is normalized already beforehand.
    return dot_product, terms


########################################################################################################################
# Functions for document to query scoring
########################################################################################################################

# Calculate the cosine similarity for a given measure type.
def query_cosine_similarity_template(query, field, measure_number):
    # First process the query.
    query_ft_data = indexer.process_text(query)
    query_ft_scores = {term: value[measure_number] for term, value in query_ft_data.items()}

    # We will keep the score of all of the papers.
    _scores = {}
    _selected_terms = {}

    # We have multiple papers. Iterate over all of them
    for paper_id, paper_tf in indexer.paper_tf_data.items():
        # Extract the tf values from the paper results.
        paper_ft_scores = {term: value[measure_number] for term, value in paper_tf[field].items()}

        # Report on the cosine similarity.
        score, terms = cosine_similarity(query_ft_scores, paper_ft_scores)

        # Save both the score and the subset of the paper term dictionary.
        _scores[paper_id] = score
        _selected_terms[paper_id] = {term: paper_ft_scores[term] for term in terms}

    # Remove scores that are zero.
    _scores = {paper_id: value for paper_id, value in _scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}

    # Sort the scores, and return.
    return sorted(_scores.items(), key=lambda x: x[1], reverse=True), _selected_terms, query_ft_scores


# Calculate the cosine similarity using the tf measure.
def query_cosine_similarity_tf(query, field):
    return query_cosine_similarity_template(query, field, 4)

    # # First process the query.
    # query_ft_data = indexer.process_text(query)
    # query_ft_scores = {term: value[4] for term, value in query_ft_data.items()}
    #
    # # We will keep the score of all of the papers.
    # _scores = {}
    # _selected_terms = {}
    #
    # # We have multiple papers. Iterate over all of them
    # for paper_id, paper_tf in indexer.paper_tf_data.items():
    #     # Extract the tf values from the paper results.
    #     paper_ft_scores = {term: value[4] for term, value in paper_tf[field].items()}
    #
    #     # Report on the cosine similarity.
    #     score, terms = cosine_similarity(query_ft_scores, paper_ft_scores)
    #
    #     # Save both the score and the subset of the paper term dictionary.
    #     _scores[paper_id] = score
    #     _selected_terms[paper_id] = {term: paper_ft_scores[term] for term in terms}
    #
    # # Remove scores that are zero.
    # _scores = {paper_id: value for paper_id, value in _scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}
    #
    # # Sort the scores, and return.
    # return sorted(_scores.items(), key=lambda x: x[1], reverse=True), _selected_terms, query_ft_scores


# Calculate the cosine similarity using the tf measure.
def query_cosine_similarity_wf(query, field):
    return query_cosine_similarity_template(query, field, 5)

########################################################################################################################
# Functions for document to document scoring
########################################################################################################################

########################################################################################################################
# Functions used to print scoring results
########################################################################################################################


def print_scoring_results(query, scoring, _selected_terms, _query_scores, top_x=10):
    print()
    print("=" * 124)
    print("query = \"" + query + "\"")
    print(min(len(scoring), top_x), "of", len(scoring), "results:")

    for i in range(0, min(len(scoring), top_x)):
        paper_id, score = scoring[i]
        print(str(i + 1) + ".\t", paper_id, "\t", database.id_to_paper[paper_id].title, score,
              _selected_terms[paper_id], _query_scores)

    print("=" * 124)


if __name__ == '__main__':
    # Initialize the indexer.
    indexer.init()

    # Calculate the cosine similarity.
    scores, selected_terms, query_scores = query_cosine_similarity_tf("chicken", "paper_text")

    # Print the results.
    print_scoring_results("chicken", scores, selected_terms, query_scores)

    # Calculate the cosine similarity.
    scores, selected_terms, query_scores = query_cosine_similarity_wf("chicken", "paper_text")

    # Print the results.
    print_scoring_results("chicken", scores, selected_terms, query_scores)
