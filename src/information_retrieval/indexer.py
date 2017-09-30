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

# Structure for the papers_term_frequency_data: (term: (tf, wf, tf.idf, wf.idf, n_tf, n_wf, n_tf.idf, n_wd.idf))
# Structure for the global_term_frequency_data: (term: (cf, df, idf))

# The fields we target in the papers.
paper_fields = ["title", "abstract", "paper_text"]


# Index all the papers that have been provided in the papers list.
def index_papers(papers):
    # Create a pool with four processes, so that we have at most 50% cpu utilization.
    with Pool(4) as pool:
        # Schedule the papers to be processed by the pool, and save the term frequency data.
        raw_paper_tf_data = pool.map(process_paper, papers)

    # A dictionary containing all terms within the paper per field.
    terms = {}

    # Map the frequency data to the paper id for easier lookups.
    paper_ft_data = {
        papers[i].id: raw_paper_tf_data[i] for i in range(0, len(papers))
    }

    # Convert all term to frequency dicts to default dicts. We do this here instead of in the pool because of errors.
    for paper_id, frequencies in paper_ft_data.items():
        a = paper_ft_data[paper_id]
        for field in paper_fields:
            # noinspection PyArgumentList
            a[field] = defaultdict(lambda: (0, 0, 0, 0, 0, 0, 0, 0), a[field])

    # Generate a list of all terms used in the papers.
    for field in paper_fields:
        # Here we keep the list of terms of the title, abstract and paper_text separately as a dictionary.
        terms[field] = set().union(*[list(fields[field]) for paper_id, fields in paper_ft_data.items()])
        print(field, len(terms[field]))

    # Calculate the collection frequency and the document frequency of every term, and safe these as a tuple.
    _global_idf_data = {}

    # Do this for every paper field separately.
    for field in paper_fields:
        _global_idf_data[field] = defaultdict(lambda: (0, 0, 0))

        # For each paper, increment the terms which occur in the paper, and sum up the frequencies for the terms.
        for paper in papers:
            for term, value in paper_ft_data[paper.id][field].items():
                x, y, z = _global_idf_data[field][term]
                _global_idf_data[field][term] = (x + value[0], y + 1, z)

        # Calculate the inverse document frequencies.
        for term, (x, y, z) in _global_idf_data[field].items():
            _global_idf_data[field][term] = (x, y, math.log2(len(papers) / y))

        # Now we can calculate the tf.idf and wf.idf measures.
        for paper in papers:
            for term, (tf, wf, _, _, n_tf, n_wf, _, _) in paper_ft_data[paper.id][field].items():
                idf = _global_idf_data[field][term][2]
                paper_ft_data[paper.id][field][term] = (tf, wf, tf * idf, wf * idf, n_tf, n_wf, 0, 0)

    # Now do some normalization in the paper_ft_data dictionary.
    for field in paper_fields:
        for paper in papers:
            data = paper_ft_data[paper.id][field]

            # We first gather the lengths of the different value types.
            tf_idf_length = math.sqrt(sum([x[2]**2 for x in data.values()]))
            wf_idf_length = math.sqrt(sum([x[3]**2 for x in data.values()]))

            # Now normalize.
            for term, (tf, wf, tf_idf, wf_idf, n_tf, n_wf, _, _) in data.items():
                data[term] = (
                    tf,
                    wf,
                    tf_idf,
                    wf_idf,
                    n_tf,
                    n_wf,
                    tf_idf / tf_idf_length,
                    wf_idf / wf_idf_length
                )

    return paper_ft_data


# Generate the index of the paper.
def process_paper(paper):
    # The paper has multiple components we might be interested in, such as the title, abstract and paper text.
    # Tokenize all of these.
    return {field: process_text(paper.__getattribute__(field)) for field in paper_fields}


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

    # Generate the frequency data.
    return generate_term_frequency_data(term_frequencies)


# Gather frequency data using the term frequency dictionary.
def generate_term_frequency_data(term_frequencies):
    # Calculate an intermediary result for the tf and wf values.
    data = {term: (frequency, 1 + math.log2(frequency), 0, 0, 0, 0, 0, 0) for term, frequency in term_frequencies.items()}

    # Calculate the length of the vector and adjust.
    tf_length = math.sqrt(sum([x[0]**2 for x in data.values()]))
    wf_length = math.sqrt(sum([x[1]**2 for x in data.values()]))

    # Now normalize.
    for term, (tf, wf, _, _, _, _, _, _) in data.items():
        data[term] = (
            tf,
            wf,
            0,
            0,
            tf / tf_length,
            wf / wf_length,
            0,
            0
        )

    return data


########################################################################################################################
# Initialization
########################################################################################################################

# The global variables we will be using between files.
paper_tf_data = None


# Initializes the indexing data.
def init():
    # Load the papers.
    database.import_papers()

    # Measure the time.
    start = time.time()

    # Index the imported papers.
    global paper_tf_data
    paper_tf_data = index_papers(database.papers)

    # Report the indexing time.
    print("Indexing took", time.time() - start, "seconds.")
