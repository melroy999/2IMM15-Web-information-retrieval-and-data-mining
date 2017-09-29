from collections import Counter, defaultdict
from multiprocessing.pool import Pool

import time

import math

from information_retrieval.analyzer import cosine_similarity_tf, cosine_similarity_tf_idf
from information_retrieval.normalizer import pre_normalization, post_normalization
from information_retrieval.tokenizer import tokenize
from import_data import database


########################################################################################################################
# Functions used during the processing of papers.
########################################################################################################################

# Structure for the papers_term_frequency_data: (term: (tf, wf, normalized_tf, normalized_wf))
# Structure for the global_term_frequency_data: (term: (cf, df, idf))

# Index all the papers that have been provided in the papers list.
def index_papers(papers):
    # Create a pool with four processes, so that we have at most 50% cpu utilization.
    with Pool(4) as pool:
        # Schedule the papers to be processed by the pool, and save the term frequency data.
        papers_term_frequency_data = pool.map(process_paper, papers)
        papers_term_frequency_data = {papers[i].id: defaultdict(lambda: (0, 0, 0, 0), papers_term_frequency_data[i]) for i in range(0, len(papers))}

    # Generate a list of all terms used in the papers.
    terms = set().union(*[list(data) for data in papers_term_frequency_data.values()])

    # Calculate the collection frequency and the document frequency of every term, and safe these as a tuple.
    global_term_frequency_data = {term: (0, 0, 0) for term in terms}
    for paper in papers:
        for term, value in papers_term_frequency_data[paper.id].items():
            x, y, z = global_term_frequency_data[term]
            global_term_frequency_data[term] = (x + value[0], y + 1, z)

    # Calculate the inverse document frequencies.
    for term, (x, y, z) in global_term_frequency_data.items():
        global_term_frequency_data[term] = (x, y, math.log2(len(papers) / y))

    # Make de dicts return 0 on default.
    global_term_frequency_data = defaultdict(lambda: (0, 0, 0), global_term_frequency_data)

    return global_term_frequency_data, papers_term_frequency_data


# Generate the index of the paper.
def process_paper(paper):
    # Process the text of the paper.
    return process_text(paper.paper_text)


# Generate the index of the text.
def process_text(text):
    # First, remove all punctuation.
    text = pre_normalization(text)

    # Next, tokenize the paper's contents.
    tokens = tokenize(text)

    # Now, do the post processing normalization.
    terms = post_normalization(tokens)

    # Create the term frequency table and the weighted term frequency table.
    term_frequencies = dict(Counter(terms))

    # Generate the frequency data and return.
    return generate_term_frequency_data(term_frequencies)


# Gather frequency data using the term frequency dictionary.
def generate_term_frequency_data(term_frequencies):
    # Calculate an intermediary result for the tf and wf values, and normalize after.
    result = {term: (frequency, 1 + math.log2(frequency), 0, 0) for term, frequency in term_frequencies.items()}

    # Calculate the vector length such that we can normalize it.
    length = math.sqrt(sum(x * x for term, (x, _, _, _) in result.items()))
    log_length = math.sqrt(sum(x * x for term, (_, x, _, _) in result.items()))

    # Normalize the vector.
    return {term: (tf, wf, tf / length, wf / log_length) for term, (tf, wf, _, _) in result.items()}


########################################################################################################################
# Functions used to print scoring results
########################################################################################################################

def print_scoring_results(id_to_imported_paper, query, scoring, top_x=10):
    print()
    print("=" * 124)
    print("query = \"" + query + "\"")
    print(min(len(scoring), top_x), "of", len(scoring), "results:")

    for i in range(0, min(len(scoring), top_x)):
        paper_id, score = scoring[i]
        print(str(i + 1) + ".\t", id_to_imported_paper[paper_id].title, score)

    print("=" * 124)


########################################################################################################################
# Main
########################################################################################################################


if __name__ == '__main__':
    # Load the papers.
    imported_papers = database.import_papers()

    # Connect the paper id to the paper.
    id_to_imported_paper = {paper.id: paper for paper in imported_papers}

    start = time.time()

    # Index the imported papers.
    collection_frequency_data, term_frequency_data = index_papers(imported_papers)

    print("Indexing time: ", time.time() - start)
    print("The frequency in paper 1: " + str(term_frequency_data[1]["the"]))
    print("Of frequency in paper 1: " + str(term_frequency_data[1]["of"]))
    print("The frequency: " + str(collection_frequency_data["the"]))
    print("Of frequency: " + str(collection_frequency_data["of"]))
    print("Neural frequency in paper 1: " + str(term_frequency_data[1]["neural"]))
    print("Neural frequency: " + str(collection_frequency_data["neural"]))
    print(len(collection_frequency_data))

    print()
    print("TF:")

    query = "This is not a very complicated query query query! chicken"
    query_term_frequency = process_text(query)
    scores = cosine_similarity_tf(collection_frequency_data, term_frequency_data, query_term_frequency)
    print_scoring_results(id_to_imported_paper, query, scores)

    query = "KWJWIWIjjaashoaishughqwfhqwphqfwbqfwbipiqwf"
    query_term_frequency = process_text(query)
    scores = cosine_similarity_tf(collection_frequency_data, term_frequency_data, query_term_frequency)
    print_scoring_results(id_to_imported_paper, query, scores)

    print()
    print("TF.IDF:")

    query = "This is not a very complicated query query query! chicken"
    query_term_frequency = process_text(query)
    scores = cosine_similarity_tf_idf(collection_frequency_data, term_frequency_data, query_term_frequency)
    print_scoring_results(id_to_imported_paper, query, scores)

    query = "KWJWIWIjjaashoaishughqwfhqwphqfwbqfwbipiqwf"
    query_term_frequency = process_text(query)
    scores = cosine_similarity_tf_idf(collection_frequency_data, term_frequency_data, query_term_frequency)
    print_scoring_results(id_to_imported_paper, query, scores)