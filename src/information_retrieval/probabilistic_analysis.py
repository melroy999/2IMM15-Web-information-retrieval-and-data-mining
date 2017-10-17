from functools import reduce
from operator import mul

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
    "None": document_probability_equal_weight,
    "Document length": document_probability_length_weight
}


# Normalize the query.
def normalize_and_tokenize_query(query, indexer):
    # Here we do need to normalize the query first...
    cleanup_instance = cleanup.get_cleanup_instance()
    altered_query = cleanup_instance.remove_control_characters(query.lower())
    altered_query = cleanup_instance.remove_punctuation(altered_query)

    # Now iterate over the query, and find the normalized value.
    normalizer = indexer.normalizer
    return [normalizer.normalize(term) for term in altered_query.split() if normalizer.is_valid_term(term)]


# Calculate the probability for a given document, using smooth mixed multinomial.
# Here low lambda is more suitable for longer queries, and high lambda is more suitable for queries that desire all
# query terms to be present.
def calc_mixture_model_probability(tokens, paper_frequencies, collection_frequencies, _lambda, p_d):
    # We will only need the term frequencies for the specific term and our total number of tokens.
    tf = paper_frequencies["tf"]
    d_l = paper_frequencies["number_of_terms"]

    cf = collection_frequencies["cf"]
    c_l = collection_frequencies["number_of_terms"]

    # Calculate the probability for this paper. here p_d is the probability that this document is chosen.
    # The latter is the probability that the query would occur according to the document model.
    return p_d(d_l, c_l) * reduce(mul, [smooth_mixed_multinomial(term, tf, d_l, cf, c_l, _lambda) for term in tokens])


# A mixture between document likelihood and collection likelihood, using a lambda as the ratio between the two.
def smooth_mixed_multinomial(term, tf, d_l, cf, c_l, _lambda):
    return _lambda * (tf[term] / d_l) + (1 - _lambda) * (cf[term] / c_l)


# Do the search for the given query, using the unigram language model.
def search(query, indexer, field, _lambda=0.0, document_probability_mode_name="None", remove_duplicates=True):
    # Now we can split the query, and we will have our tokens.
    query_tokens = normalize_and_tokenize_query(query, indexer)

    # Remove duplicates if required.
    if remove_duplicates:
        query_tokens = set(query_tokens)

    # The query can end up empty because of tokenization. So throw an exception of this is the case.
    if len(query_tokens) == 0:
        raise EmptyQueryException()

    # We will need the same collection frequencies for all calculations.
    collection_frequencies = indexer.results["collection"][field]

    # Set the document probability mode.
    p_d = document_probability_modes[document_probability_mode_name]

    # Check all papers.
    chances = {}
    for paper_id, paper_frequencies in indexer.results["papers"][field].items():
        # Calculate the probability.
        chances[paper_id] = calc_mixture_model_probability(query_tokens, paper_frequencies, collection_frequencies,
                                                           _lambda, p_d)

    # Now we can find the papers with the highest chance.
    return sorted(chances.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    _indexer = Indexer()
    _indexer.index_corpus("None", True)
    result = search("Neural, Neural", _indexer, "paper_text", _lambda=0.0)
    print(result[:10])
    result = search("Neural, Neural", _indexer, "paper_text", _lambda=0.5)
    print(result[:10])
    result = search("Neural, Neural", _indexer, "paper_text", _lambda=1.0)
    print(result[:10])
