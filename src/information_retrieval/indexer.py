from collections import Counter, defaultdict
from multiprocessing.pool import Pool

import time

import math

import gc

from information_retrieval import lemmatizer, stemmer
from information_retrieval.normalizer import *
from information_retrieval.stemmer import *
from information_retrieval.tokenizer import tokenize
from import_data import database

# The list of stemming options.
# noinspection PyArgumentList
stemming_options = {
    "nltk porter": execute_nltk_porter_stemmer,
    "nltk lancaster": execute_nltk_lancaster_stemmer,
    "nltk snowball": execute_nltk_snowball_stemmer,
    "porter": execute_porter_stemmer,
    "porter 2": execute_porter_2_stemmer,
    "wordnet lemmatizer": execute_nltk_wordnet_lemmatizer
}


class Indexer(object):
    ####################################################################################################################
    # Functions used during the processing of papers.
    ####################################################################################################################

    # Structure for the papers_term_frequency_data: (term: (tf, wf, tf.idf, wf.idf, n_tf, n_wf, n_tf.idf, n_wd.idf))
    # Structure for the global_term_frequency_data: (term: (cf, df, idf))

    # The global variables we will be using between files.
    paper_tf_data = None

    # The fields we target in the papers.
    paper_fields = ["title", "abstract", "paper_text"]

    # The stemming mode used.
    stemming_mode = execute_nltk_wordnet_lemmatizer

    # Whether we use stop words or not.
    use_stopwords = True

    # The status bar reference.
    status_bar = None

    # Index all the papers that have been provided in the papers list.
    def index_papers(self, papers):
        # Create a pool with four processes, so that we have at most 50% cpu utilization.
        self.update_status("Gathering paper term frequency data...")
        with Pool(4) as pool:
            # Schedule the papers to be processed by the pool, and save the term frequency data.
            raw_paper_tf_data = pool.map(self.process_paper, papers)

        # A dictionary containing all terms within the paper per field.
        terms = {}

        # Map the frequency data to the paper id for easier lookups.
        self.update_status("Mapping paper id to term frequency data...")
        paper_ft_data = {
            papers[i].id: raw_paper_tf_data[i] for i in range(0, len(papers))
        }

        # Convert all term to frequency dicts to default dicts.
        self.update_status("Setting term frequency default vectors...")
        for paper_id, frequencies in paper_ft_data.items():
            a = paper_ft_data[paper_id]
            for field in self.paper_fields:
                # noinspection PyArgumentList
                a[field] = defaultdict(lambda: (0, 0, 0, 0, 0, 0, 0, 0), a[field])

        # Generate a list of all terms used in the papers.
        self.update_status("Gathering term occurrences per paper field...")
        print("Gathering term lists:")
        for field in self.paper_fields:
            # Here we keep the list of terms of the title, abstract and paper_text separately as a dictionary.
            terms[field] = set().union(*[list(fields[field]) for paper_id, fields in paper_ft_data.items()])
            print("- Encountered " + str(len(terms[field])) + " unique terms for field \"" + field + "\".")
        print()

        # Calculate the collection frequency and the document frequency of every term, and safe these as a tuple.
        _global_idf_data = {}

        # Do this for every paper field separately.
        self.update_status("Calculate collection frequency and document frequency...")
        for field in self.paper_fields:
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
        self.update_status("Normalize document vectors...")
        for field in self.paper_fields:
            for paper in papers:
                data = paper_ft_data[paper.id][field]

                # We first gather the lengths of the different value types.
                tf_idf_length = math.sqrt(sum([x[2] ** 2 for x in data.values()]))
                wf_idf_length = math.sqrt(sum([x[3] ** 2 for x in data.values()]))

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

    # Update the status in the gui, if possible.
    def update_status(self, status):
        try:
            self.status_bar(status)
        except TypeError:
            pass

    # Generate the index of the paper.
    def process_paper(self, paper):
        # The paper has multiple components we might be interested in, such as the title, abstract and paper text.
        # Tokenize all of these.
        return {field: self.process_text(paper.__getattribute__(field)) for field in self.paper_fields}

    # Generate the index of the text.
    def process_text(self, text):
        # First, remove all punctuation.
        text = pre_normalization(text)

        # Next, tokenize the paper's contents.
        tokens = tokenize(text)

        # Now, do the post processing normalization.
        terms = post_normalization(tokens, normalizer=lambda x: self.stemming_mode(x), use_stopwords=self.use_stopwords)

        # Create the term frequency table and the weighted term frequency table.
        term_frequencies = dict(Counter(terms))

        # Generate the frequency data.
        return self.generate_term_frequency_data(term_frequencies)

    # Gather frequency data using the term frequency dictionary.
    @staticmethod
    def generate_term_frequency_data(term_frequencies):
        # Calculate an intermediary result for the tf and wf values.
        data = {term: (frequency, 1 + math.log2(frequency), 0, 0, 0, 0, 0, 0) for term, frequency in
                term_frequencies.items()}

        # Calculate the length of the vector and adjust.
        tf_length = math.sqrt(sum([x[0] ** 2 for x in data.values()]))
        wf_length = math.sqrt(sum([x[1] ** 2 for x in data.values()]))

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

    ####################################################################################################################
    # Initialization
    ####################################################################################################################

    # Reset all global data we have.
    def reset(self):
        # Reset paper_tf_data
        self.paper_tf_data = None

        # Reset the lemmatizer and stemmer helper dictionaries.
        lemmatizer.term_to_lemma = {}
        stemmer.term_to_stem = {}

        # Do garbage collection.
        gc.collect()

    # Start the indexing for the papers.
    def index(self, stemming_mode, use_stopwords, status_bar):
        # Reset all data gathered in a previous run.
        self.reset()

        # Set the modes.
        self.stemming_mode = stemming_options[stemming_mode]
        self.use_stopwords = use_stopwords
        self.status_bar = status_bar

        # Measure the time.
        start = time.time()

        # Index the imported papers.
        self.paper_tf_data = self.index_papers(database.papers)

        # Report the indexing time.# Report finishing indexing.
        self.update_status("Finished indexing")
        print("Indexing took", time.time() - start, "seconds.")
        print("Finished indexing")
        print()

    # Initializes the indexer.
    def __init__(self):
        # Load the papers.
        database.import_papers()
