from collections import Counter, defaultdict

import time
from functools import partial
from multiprocessing.pool import Pool

import math

import import_data.database as database
from information_retrieval.normalizer import *

database.import_papers()
papers = database.papers


# Process the term frequencies of the terms found in the paper.
def process_paper(paper, normalizer, field):
    lowercase = paper.__getattribute__(field).lower()
    no_punctuation = normalizer.remove_punctuation(lowercase)
    tokens = no_punctuation.split()

    # Generate the tf frequencies, with stemming and stopwords.
    tf = calculate_tf(tokens, normalizer)

    # Generate the wf frequencies and calculate the length of the two vectors.
    tf_length, wf, wf_length = calculate_wf_and_lengths(tf)

    return tf, wf, math.sqrt(tf_length), math.sqrt(wf_length)


# Calculate the term frequencies of the tokens, with the given normalizer.
def calculate_tf(tokens, _normalizer):
    tf = defaultdict(int)
    for term, value in Counter(tokens).items():
        if english_stopwords.__contains__(term):
            continue
        norm_term = _normalizer.normalize(term)
        tf[norm_term] = tf[norm_term] + value
    return tf


# Calculate the weighted term frequencies, next to the document vector lengths.
def calculate_wf_and_lengths(tf):
    tf_length = 0
    wf_length = 0
    wf = defaultdict(int)
    for term, value in tf.items():
        wf[term] = 1 + math.log2(value)
        tf_length += value ** 2
        wf_length += wf[term] ** 2
    return tf_length, wf, wf_length


# Index a certain field for all the papers, with multiprocessing when defined.
def index(field, _normalizer, multiprocessing=True):
    if multiprocessing:
        with Pool(4) as pool:
            # Schedule the papers to be processed by the pool, and save the term frequency data.
            paper_term_frequencies = pool.map(partial(process_paper, normalizer=_normalizer, field=field), papers[:])
    else:
        paper_term_frequencies = []
        for paper in papers:
            paper_term_frequencies.append(process_paper(paper, _normalizer, field))

    # Calculate the collective statistics, such as the cf and df measures.
    idf_collection = calculate_cf_df(paper_term_frequencies)

    # Calculate the inverse document frequency.
    idf_collection, idf_collection_length = calculate_idf_and_idf_length(idf_collection)

    return paper_term_frequencies, idf_collection, idf_collection_length


# Calculate the collection frequency and the document frequency.
def calculate_cf_df(paper_term_frequencies):
    idf_collection = defaultdict(lambda: (0, 0, 0))
    for paper_tf, _, _, _ in paper_term_frequencies:
        for term, value in paper_tf.items():
            x, y, _ = idf_collection[term]
            idf_collection[term] = (x + value, y + 1, 0)

    return idf_collection


# Calculate the inverse document frequency and the idf document vector length.
def calculate_idf_and_idf_length(idf_collection):
    paper_log = math.log2(len(papers))
    idf_collection_length = 0
    for term, (cf, df, _) in idf_collection.items():
        idf_collection[term] = (cf, df, paper_log - math.log2(df))
        idf_collection_length += idf_collection[term][2] ** 2
    return idf_collection, idf_collection_length


# Calculate a full index of the papers.
# This includes the fields: paper_text, abstract, title
def full_index():
    normalizer = Normalizer(use_stopwords, "nltk_porter_stemmer")
    start = time.time()
    index("paper_text", normalizer, True)
    index("abstract", normalizer, False)
    index("title", normalizer, False)
    normalizer.create_table_file()
    timers.append(time.time() - start)
    print(time.time() - start)

if __name__ == '__main__':
    n = 2
    timers = []
    for i in range(0, n):
        full_index()

    print(min(timers), sum(timers) / n, max(timers))