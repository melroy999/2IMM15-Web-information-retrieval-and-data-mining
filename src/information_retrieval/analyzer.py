import math

from import_data import database


# The search modes we have, such as tf, wf, tf.idf and wf.idf scoring and which id we can access the variables from.
# noinspection PyArgumentList
scoring_measure_ids = {
    "tf": 4,
    "wf": 5,
    "tf.idf": 6,
    "wf.idf": 7
}

# Amount of results we can show.
results_to_show = [10, 20, 50, 100, 1000, 10000]


class Analyzer(object):
    # The indexer to take the data from and the query we will execute.
    def __init__(self, indexer, query, field):
        self.indexer = indexer
        self.query = query
        self.field = field

    ####################################################################################################################
    # Generic functions used during all similarity measures.
    ####################################################################################################################

    # Calculate the vector length of a term frequency table.
    @staticmethod
    def calc_vector_length(terms, term_to_score):
        return math.sqrt(sum([term_to_score[term] ** 2 for term in terms]))

    # Calculate the dot product of the two documents.
    @staticmethod
    def calc_dot_product(terms, term_to_score_doc_1, term_to_score_doc_2):
        return sum(term_to_score_doc_1[term] * term_to_score_doc_2[term] for term in terms)

    # Calculate the cosine similarity of two data sets.
    def cosine_similarity(self, term_to_score_doc_1, term_to_score_doc_2, intersection=True):
        # We first need to find the complete set of terms.
        # It holds for all measures that missing terms get the score zero, thus we can take the intersection of sets.
        # Just to be sure we made this toggle-able.
        if intersection:
            terms = set(term_to_score_doc_1).intersection(term_to_score_doc_2)
        else:
            terms = set(term_to_score_doc_1).union(term_to_score_doc_2)

        # Calculate the dot product and vector lengths of the two data sets.
        dot_product = self.calc_dot_product(terms, term_to_score_doc_1, term_to_score_doc_2)

        # We assume that the input is normalized already beforehand.
        return dot_product, terms

    ####################################################################################################################
    # Functions for document to query scoring
    ####################################################################################################################

    # Calculate the cosine similarity for a given measure type.
    def query_cosine_similarity_template(self, query, field, measure_number):
        # First process the query.
        query_ft_data = self.indexer.process_text(query)
        query_ft_scores = {term: value[measure_number] for term, value in query_ft_data.items()}

        # We will keep the score of all of the papers.
        _scores = {}
        _selected_terms = {}

        # We have multiple papers. Iterate over all of them
        for paper_id, paper_tf in self.indexer.paper_tf_data.items():
            # Extract the tf values from the paper results.
            paper_ft_scores = {term: value[measure_number] for term, value in paper_tf[field].items()}

            # Report on the cosine similarity.
            score, terms = self.cosine_similarity(query_ft_scores, paper_ft_scores)

            # Save both the score and the subset of the paper term dictionary.
            _scores[paper_id] = score
            _selected_terms[paper_id] = {term: paper_ft_scores[term] for term in terms}

        # Remove scores that are zero.
        _scores = {paper_id: value for paper_id, value in _scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}

        # Sort the scores, and return.
        return sorted(_scores.items(), key=lambda x: x[1], reverse=True), _selected_terms, query_ft_scores

    # Calculate the cosine similarity using the tf measure.
    def query_cosine_similarity_tf(self, query, field):
        return self.query_cosine_similarity_template(query, field, 4)

    # Calculate the cosine similarity using the tf measure.
    def query_cosine_similarity_wf(self, query, field):
        return self.query_cosine_similarity_template(query, field, 5)

    ####################################################################################################################
    # Functions for document to document scoring
    ####################################################################################################################

    # Calculate the cosine similarity for a given measure type.
    def document_cosine_similarity_template(self, paper_id, field, measure_number):
        # First process the query.
        query_ft_scores = {term: value[measure_number] for term, value in self.indexer.paper_tf_data[paper_id][field].items()}

        # We will keep the score of all of the papers.
        _scores = {}
        _selected_terms = {}

        # We have multiple papers. Iterate over all of them
        for paper_id, paper_tf in self.indexer.paper_tf_data.items():
            # Extract the tf values from the paper results.
            paper_ft_scores = {term: value[measure_number] for term, value in paper_tf[field].items()}

            # Report on the cosine similarity.
            score, terms = self.cosine_similarity(query_ft_scores, paper_ft_scores)

            # Save both the score and the subset of the paper term dictionary.
            _scores[paper_id] = score
            _selected_terms[paper_id] = {term: paper_ft_scores[term] for term in terms}

        # Remove scores that are zero.
        _scores = {paper_id: value for paper_id, value in _scores.items() if not math.isclose(value, 0.0, abs_tol=1e-19)}

        # Sort the scores, and return.
        return sorted(_scores.items(), key=lambda x: x[1], reverse=True), _selected_terms, query_ft_scores

    # Calculate the cosine similarity using the tf measure.
    def document_cosine_similarity_tf(self, paper_id, field):
        return self.document_cosine_similarity_template(paper_id, field, 4)

    # Calculate the cosine similarity using the tf measure.
    def document_cosine_similarity_wf(self, paper_id, field):
        return self.document_cosine_similarity_template(paper_id, field, 5)

    # Calculate the cosine similarity using the tf measure.
    def document_cosine_similarity_tf_idf(self, paper_id, field):
        return self.document_cosine_similarity_template(paper_id, field, 6)

    # Calculate the cosine similarity using the tf measure.
    def document_cosine_similarity_wf_idf(self, paper_id, field):
        return self.document_cosine_similarity_template(paper_id, field, 7)

    ####################################################################################################################
    # Functions used to print scoring results
    ####################################################################################################################

    @staticmethod
    def print_scoring_results(query, scoring, _selected_terms, _query_scores, top_x=10, report_dataset=False):
        print()
        print("=" * 70)
        print("query = \"" + query + "\"")
        print(min(len(scoring), top_x), "of", len(scoring), "results:")

        for i in range(0, min(len(scoring), top_x)):
            paper_id, score = scoring[i]
            if report_dataset:
                print(str(i + 1) + ".\t", paper_id, "\t", '%0.8f' % score, "\t", database.id_to_paper[paper_id].title,
                      _selected_terms[paper_id], _query_scores)
            else:
                print(str(i + 1) + ".\t", paper_id, "\t", '%0.8f' % score, "\t", database.id_to_paper[paper_id].title)

        print("=" * 70)

    def search(self, is_document, scoring_mode, top_x):
        if is_document:
            # This will be tricky, first try to convert the query to an int to see if we have an id as input.
            try:
                paper_id = int(self.query)
            except ValueError:
                # If it is not, we know that we have a paper title. Find the paper that corresponds most.
                scores, _, _ = self.query_cosine_similarity_template(self.query, "title", 4)

                # Try to take the best scoring paper as target. This can be empty however!
                try:
                    paper_id = scores[0][0]
                except KeyError:
                    print("Paper not found. Please try again!")
                    return

            # Report on what we found.
            print("Target paper: #" + str(paper_id) + " \"" + database.id_to_paper[paper_id].title + "\"")

            # Handle the query as a paper id.
            scores, selected_terms, query_scores = self.document_cosine_similarity_template(paper_id, self.field, scoring_measure_ids[scoring_mode])
        else:
            if scoring_measure_ids[scoring_mode] > 5:
                print("Error: the scoring mode \"" + scoring_mode + "\" is not supported for normal queries.")
                return
            scores, selected_terms, query_scores = self.query_cosine_similarity_template(self.query, self.field, scoring_measure_ids[scoring_mode])

        self.print_scoring_results(self.query, scores, selected_terms, query_scores, top_x)
