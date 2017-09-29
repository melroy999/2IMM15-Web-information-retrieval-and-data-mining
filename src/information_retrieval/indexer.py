from collections import Counter, defaultdict
from multiprocessing.pool import Pool

import time

import math

from information_retrieval.normalizer import pre_normalization, post_normalization
from information_retrieval.tokenizer import tokenize
from import_data import database


########################################################################################################################
# Functions used during the processing of papers.
########################################################################################################################

# Index all the papers that have been provided in the papers list.
def index_papers(papers):
    # Create a pool with four processes, so that we have at most 50% cpu utilization.
    with Pool(4) as pool:
        # Schedule the papers to be processed by the pool, and save the term frequency data.
        papers_term_frequency_data = pool.map(process_paper, papers)
        papers_term_frequency_data = {papers[i].id: defaultdict(lambda: 0, papers_term_frequency_data[i]) for i in range(0, len(papers))}

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
    global_term_frequency_data = defaultdict(lambda: 0, global_term_frequency_data)

    return global_term_frequency_data, papers_term_frequency_data


# Generate the index of the paper object.
def process_paper(paper):
    # First, remove all punctuation.
    text = pre_normalization(paper.paper_text)

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
    # Pre-calculate the weighted frequency of each term, and store it as a tuple together with the term frequency.
    return {term: (frequency, 1 + math.log2(frequency)) for term, frequency in term_frequencies.items()}

########################################################################################################################
# Main
########################################################################################################################


if __name__ == '__main__':
    # Load the papers.
    imported_papers = database.import_papers()

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
