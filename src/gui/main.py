import threading
import traceback
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import time

import information_retrieval.boolean_analysis as ba
import information_retrieval.vector_space_analysis as vsa
from import_data import database
from classification import classification
from information_retrieval.indexer import paper_fields, Indexer
from information_retrieval.normalizer import name_to_normalizer

from import_data.crawler_database import get_info
from information_retrieval.probabilistic_analysis import search_modes, document_probability_modes, okapi_idf_modes, \
    ProbabilisticAnalysis

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import information_retrieval.clustering as KMclus
import information_retrieval.clustering_dbscan as DBclus

# Amount of results we can show.
results_to_show = [10, 20, 50, 100, 1000, 10000]


# The part of the GUI which handles indexing settings and functions.
class IndexFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(2, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Now, make drop down menus for the type of stemming and the search method in a different frame.
        self.stemming_var = StringVar(self)
        stemming_choices = [stemmer for stemmer in name_to_normalizer.keys()]
        self.stemming_var.set(stemming_choices[-1])
        self.stemming_label = ttk.Label(self, text="Stemmer:")
        self.stemming_field = ttk.Combobox(self, values=stemming_choices, textvariable=self.stemming_var,
                                           state="readonly")

        self.stemming_field.config(width=22)
        self.stemming_label.grid(row=0, column=0, sticky=W)
        self.stemming_field.grid(row=0, column=1, sticky=W, padx=10)

        # Now add some checkboxes for other options that the indexer provides.
        self.find_stop_words = ttk.Checkbutton(self, text="Enable stopwords")
        self.find_stop_words.state(['!alternate'])
        self.find_stop_words.state(['selected'])
        self.find_stop_words.grid(row=0, column=2, sticky=W)

        # Create a button to start indexing.
        self.indexing_button = ttk.Button(self, text="Index papers", command=lambda: self.start_indexing(), width=20)
        self.indexing_button.grid(row=0, column=3, sticky=E)

    def start_indexing(self):
        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the selected stemming mode.
        stemming_mode = gui.index_frame.stemming_var.get()

        # Get whether we want to use stopwords or not.
        use_stopwords = gui.index_frame.find_stop_words.instate(['selected'])

        # Change the status.
        update_status("Indexing...")
        print("=== INDEXER ===")
        print("Starting indexing with the following settings: ")
        print("- Stemming/lemmatizing:", stemming_mode)
        print("- Use stopwords: ", use_stopwords)
        print()

        # Initialize the indexer.
        def runner():
            indexer.index_corpus(stemming_mode, use_stopwords)
            self.finish_indexing()

        t = threading.Thread(target=runner)
        t.start()

    @staticmethod
    def finish_indexing():
        # Enable the index and search buttons.
        enable_search_buttons()
        update_status("Finished indexing")


# The part of the GUI which handles boolean query settings and functions.
class BooleanQueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = ttk.Label(self, text="Query: ")
        self.query_field = ttk.Entry(self)
        self.query_button = ttk.Button(self, text="Search", command=lambda: self.start_analyzing(), width=20,
                                       state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, columnspan=5, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=6, sticky=E)

        # Advanced options for the querying.
        self.options_frame = Frame(self)
        self.options_frame.grid_columnconfigure(5, weight=1)
        self.options_frame.grid(row=1, column=1, columnspan=5, sticky=W + E, pady=(3, 0))

        self.target_field = CompoundComboBox(self.options_frame, [field for field in paper_fields],
                                             "Default target field: ", -1)
        self.target_field.grid(row=1, column=1, sticky=W, padx=(10, 0))

        self.result_count_var = IntVar(self)
        result_count_choices = [results for results in results_to_show]
        self.result_count_var.set(result_count_choices[0])
        self.result_count_label = ttk.Label(self.options_frame, text="#Results: ")
        self.result_count_field = ttk.Combobox(self.options_frame, values=result_count_choices,
                                               textvariable=self.result_count_var,
                                               state="readonly")

        self.result_count_field.config(width=22)
        self.result_count_label.grid(row=1, column=2, sticky=W)
        self.result_count_field.grid(row=1, column=3, sticky=W, padx=10)

        self.show_crawled_data = ttk.Checkbutton(self, text="Show crawled data")
        self.show_crawled_data.state(['!alternate'])
        self.show_crawled_data.state(['selected'])
        self.show_crawled_data.grid(row=1, column=5, sticky=W, padx=(550, 0))

    # Start analyzing the query and find results.
    def start_analyzing(self):
        # Get the query.
        query = self.query_field.get().strip()

        # Make sure we are not doing an empty query...
        if query == '':
            return

        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the query mode and whether we are searching for a document or not.
        target_default_field = self.target_field.get()
        result_count = self.result_count_var.get()

        # Change the status.
        update_status("Searching...")
        print("=== BOOLEAN ANALYZER ===")
        print("Starting boolean search with the following settings: ")
        print("- Target field:", target_default_field)
        print("- Number of results:", result_count)
        print()

        # Initialize the analyzer.
        def runner():
            # Calculate the scores.
            # query, indexer, field, scoring_measure="tf", similar_document_search=False
            try:
                start = time.time()
                results = ba.search(query, indexer, target_default_field)

                # The time at which we are done.
                print("Running time:", time.time() - start)

                # Print the results.
                use_crawler = self.show_crawled_data.instate(['selected'])
                print_results(query, results, use_crawler, has_score=False, top_x=result_count)
            except Exception as e:
                print("Received an invalid query. Keep in mind that the boolean analysis only supports single word "
                      "queries (words containing a hyphen are considered multiple words).")
                print(e)
                traceback.print_tb(e.__traceback__)

            # Finish the analyzing process.
            self.finish_analyzing()

        t = threading.Thread(target=runner)
        t.start()

    # Finish analyzing and report the results.
    @staticmethod
    def finish_analyzing():
        # Enable the index and search buttons.
        enable_search_buttons()

        # Change the status.
        update_status("Finished searching")
        print()


# The part of the GUI which handles vector space query settings and functions.
class VectorSpaceQueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = ttk.Label(self, text="Query: ")
        self.query_field = ttk.Entry(self)
        self.query_button = ttk.Button(self, text="Search", command=lambda: self.start_analyzing(),
                                       width=20, state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, columnspan=5, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=6, sticky=E)

        # Advanced options for the querying.
        self.options_frame = Frame(self)
        self.options_frame.grid_columnconfigure(5, weight=1)
        self.options_frame.grid(row=1, column=1, columnspan=5, sticky=W + E, pady=(3, 0))

        self.search_method = CompoundComboBox(self.options_frame,
                                              [scoring_mode for scoring_mode in vsa.scoring_measures],
                                              "Search measurement: ")
        self.search_method.grid(row=1, column=1, sticky=W, padx=(10, 0))

        self.similarity_measure = CompoundComboBox(self.options_frame,
                                                   [measure for measure in vsa.similarity_measures],
                                                   "Similarity measure: ", 2)
        self.similarity_measure.grid(row=1, column=2, sticky=W)

        self.target_field = CompoundComboBox(self.options_frame, [field for field in paper_fields],
                                             "Paper field: ", -1)
        self.target_field.grid(row=1, column=3, sticky=W)

        self.result_count_var = IntVar(self)
        result_count_choices = [results for results in results_to_show]
        self.result_count_var.set(result_count_choices[0])
        self.result_count_label = ttk.Label(self.options_frame, text="#Results: ")
        self.result_count_field = ttk.Combobox(self.options_frame, values=result_count_choices,
                                               textvariable=self.result_count_var,
                                               state="readonly")

        self.result_count_field.config(width=22)
        self.result_count_label.grid(row=1, column=4, sticky=W)
        self.result_count_field.grid(row=1, column=5, sticky=W, padx=10)

        self.find_comparable_papers = ttk.Checkbutton(self.options_frame, text="Find similar papers based on title")
        self.find_comparable_papers.state(['!alternate'])
        self.find_comparable_papers.grid(row=1, column=5, sticky=E, padx=(0, 10))

        self.show_crawled_data = ttk.Checkbutton(self, text="Show crawled data")
        self.show_crawled_data.state(['!alternate'])
        self.show_crawled_data.state(['selected'])
        self.show_crawled_data.grid(row=2, column=5, sticky=W, padx=(0, 0))

    # Start analyzing the query and find results.
    def start_analyzing(self):
        # Get the query.
        query = self.query_field.get().strip()

        # Make sure we are not doing an empty query...
        if query == '':
            return

        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the query mode and whether we are searching for a document or not.
        find_similar_documents = self.find_comparable_papers.instate(['selected'])
        query_score_mode = self.search_method.get()
        similarity_measure = self.similarity_measure.get()
        target_field = self.target_field.get()
        result_count = self.result_count_var.get()

        # Change the status.
        update_status("Searching...")
        print("=== VECTOR-SPACE ANALYZER ===")
        print("Starting vector space search with the following settings: ")
        print("- Scoring mode:", query_score_mode)
        print("- Similarity measure:", similarity_measure)
        print("- Search for similar document:", find_similar_documents)
        print("- Target field:", target_field)
        print("- Number of results:", result_count)
        print()

        # Initialize the analyzer.
        def runner():
            # Calculate the scores.
            # query, indexer, field, scoring_measure="tf", similar_document_search=False
            try:
                start = time.time()
                scores = vsa.search(query, indexer, target_field, query_score_mode, find_similar_documents,
                                    similarity_measure)

                # The time at which we are done.
                print("Running time:", time.time() - start)

                if scores is not None:
                    # Print the scores.
                    use_crawler = self.show_crawled_data.instate(['selected'])
                    print_results(query, scores, use_crawler, has_score=True, top_x=result_count)
            except vsa.EmptyQueryException:
                print("Query is empty after normalization, please change the query.")

            # Finish the analyzing process.
            self.finish_analyzing()

        t = threading.Thread(target=runner)
        t.start()

    # Finish analyzing and report the results.
    @staticmethod
    def finish_analyzing():
        # Enable the index and search buttons.
        enable_search_buttons()

        # Change the status.
        update_status("Finished searching")
        print()

    # @staticmethod
    # def print_results(query, scores, top_x=10):
    #     if len(scores) == 0:
    #         print("No results found for query \"" + query + "\"!")
    #         return
    #
    #     print("query = \"" + query + "\"")
    #     print(min(len(scores), top_x), "of", len(scores), "results:")
    #
    #     for i in range(0, min(len(scores), top_x)):
    #         paper_id, score = scores[i]
    #         title = database.paper_id_to_paper[paper_id].stored_title
    #         crawled_data = ""
    #         if gui.vector_space_query_frame.show_crawled_data.instate(['selected']):
    #             crawled_data = get_info(title, indent=True)
    #         print(str(i + 1) + ".\t", paper_id, "\t", '%0.8f' % score, "\t", title)
    #         print(crawled_data, "\n")


# A helper class to make combo boxes a bit more organized.
class CompoundComboBox(Frame):
    def __init__(self, master, options, label_text, default_option=0):
        super().__init__(master)

        self.var = StringVar(self)
        options = options
        self.var.set(options[default_option])
        self.label = ttk.Label(self, text=label_text)
        self.combobox = ttk.Combobox(self, values=options,
                                     textvariable=self.var,
                                     state="readonly")
        self.combobox.config(width=22)
        self.label.grid(row=0, column=0, sticky=W, padx=(0, 5))
        self.combobox.grid(row=0, column=1, sticky=W, padx=(0, 20))

    def get(self):
        return self.var.get()


# A helper class for sliders displaying their current value.
class CompoundSlider(Frame):
    @staticmethod
    def do_nothing_with_value(value):
        pass

    def update_slider_value(self, value):
        # Round to one digit.
        value = round(2 * float(value), 1) / 2

        self.slider_value_display.configure(text="%.2f" % value)

        # We have to disable the command for a moment.
        self.slider.configure(command=CompoundSlider.do_nothing_with_value)
        self.slider.set(value)
        self.slider.configure(command=self.update_slider_value)

    def __init__(self, master, from_, to, value, label):
        super().__init__(master)
        # self.pack(fill=X, expand=0)

        self.slider_label = ttk.Label(self, text=label)
        self.slider_label.grid(row=0, column=0, sticky=W)
        self.slider = ttk.Scale(self, from_=from_, to=to, value=value, command=self.update_slider_value)
        self.slider.grid(row=0, column=1, sticky=W)
        self.slider_value_display = ttk.Label(self, text="%.2f" % self.slider.get())
        self.slider_value_display.grid(row=0, column=2, sticky=W)

    def get(self):
        return self.slider.get()


# The part of the GUI which handles probabilistic query settings and functions.
class ProbabilisticQueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(50, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = ttk.Label(self, text="Query: ")
        self.query_field = ttk.Entry(self)
        self.query_button = ttk.Button(self, text="Search", command=lambda: self.start_analyzing(),
                                       width=20, state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, columnspan=50, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=51, sticky=E)

        # Advanced options for the querying.
        self.options_frame = Frame(self)
        self.options_frame.grid_columnconfigure(5, weight=1)
        self.options_frame.grid(row=1, column=1, columnspan=50, sticky=W + E, pady=(3, 0))

        self.search_mode = CompoundComboBox(self.options_frame, [results for results in search_modes],
                                            "Search mode: ")
        self.search_mode.grid(row=1, column=1, sticky=W, padx=(10, 0))

        self.probability_mode = CompoundComboBox(self.options_frame,
                                                 [results for results in document_probability_modes], "p(d): ")
        self.probability_mode.grid(row=1, column=2, sticky=W)

        self.idf_mode = CompoundComboBox(self.options_frame, [results for results in okapi_idf_modes],
                                         "Okapi idf mode: ", 1)
        self.idf_mode.grid(row=1, column=3, sticky=W)

        self.target_field = CompoundComboBox(self.options_frame, [results for results in paper_fields],
                                             "Target field: ", 2)
        self.target_field.grid(row=1, column=4, sticky=W)

        self.remove_duplicate_terms = ttk.Checkbutton(self.options_frame, text="Remove duplicate terms in query")
        self.remove_duplicate_terms.state(['!alternate'])
        self.remove_duplicate_terms.grid(row=1, column=5, sticky=E, padx=10)

        # Create a new frame for better looks.
        self.slider_frame = Frame(self)
        self.slider_frame.grid_columnconfigure(5, weight=1)
        self.slider_frame.grid(row=2, column=1, columnspan=50, sticky=W + E, pady=(5, 0))

        self.lambda_slider = CompoundSlider(self.slider_frame, 0.0, 1.0, 1.0, u"\u03BB: ")
        self.lambda_slider.grid(row=0, column=0, sticky=W, padx=10)

        self.k_1_slider = CompoundSlider(self.slider_frame, 1.2, 2.0, 1.5, u"k\u2081: ")
        self.k_1_slider.grid(row=0, column=1, sticky=W, padx=10)

        self.b_slider = CompoundSlider(self.slider_frame, 0.5, 1.0, 0.75, "b: ")
        self.b_slider.grid(row=0, column=2, sticky=W, padx=10)

        self.delta_slider = CompoundSlider(self.slider_frame, 0.0, 2.0, 1.0, u"\u03B4: ")
        self.delta_slider.grid(row=0, column=3, sticky=W, padx=10)

        self.epsilon_slider = CompoundSlider(self.slider_frame, 0.0, 1.0, 0.1, u"\u03B5: ")
        self.epsilon_slider.grid(row=0, column=4, sticky=W, padx=10)

        self.result_count_var = IntVar(self)
        result_count_choices = [results for results in results_to_show]
        self.result_count_var.set(result_count_choices[0])
        self.result_count_label = ttk.Label(self.slider_frame, text="#Results: ")
        self.result_count_field = ttk.Combobox(self.slider_frame, values=result_count_choices,
                                               textvariable=self.result_count_var, state="readonly")

        self.result_count_field.config(width=22)
        self.result_count_label.grid(row=0, column=5, sticky=E)
        self.result_count_field.grid(row=0, column=6, sticky=E, padx=10)

        self.show_crawled_data = ttk.Checkbutton(self, text="Show crawled data")
        self.show_crawled_data.state(['!alternate'])
        self.show_crawled_data.state(['selected'])
        self.show_crawled_data.grid(row=3, column=1, sticky=W, padx=(0, 0))

        self.probability_searcher = ProbabilisticAnalysis()

    # Start analyzing the query and find results.
    def start_analyzing(self):
        # Get the query.
        query = self.query_field.get().strip()

        # Make sure we are not doing an empty query...
        if query == '':
            return

        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the query mode and whether we are searching for a document or not.
        target_field = self.target_field.get()
        search_mode_name = self.search_mode.get()
        document_probability_mode_name = self.probability_mode.get()
        okapi_idf_mode_name = self.idf_mode.get()
        remove_duplicates = self.remove_duplicate_terms.instate(['selected'])
        _lambda = self.lambda_slider.get()
        k_1 = self.k_1_slider.get()
        b = self.b_slider.get()
        delta = self.delta_slider.get()
        epsilon = self.epsilon_slider.get()

        result_count = self.result_count_var.get()

        # Change the status.
        update_status("Searching...")
        print("=== PROBABILISTIC ANALYZER ===")

        # Initialize the analyzer.
        def runner():
            # Calculate the scores.
            # query, indexer, field, scoring_measure="tf", similar_document_search=False
            try:
                start = time.time()
                # Here we should use the instance we created earlier.
                scores = self.probability_searcher.search(query, indexer, target_field, search_mode_name,
                                                          document_probability_mode_name, okapi_idf_mode_name,
                                                          remove_duplicates, _lambda, k_1, b, delta, epsilon)

                # The time at which we are done.
                print("Running time:", time.time() - start)

                if scores is not None:
                    # Print the scores.
                    use_crawler = self.show_crawled_data.instate(['selected'])
                    print_results(query, scores, use_crawler, has_score=True, top_x=result_count, score_format="%0.8e")
            except vsa.EmptyQueryException:
                print("Query is empty after normalization, please change the query.")

            # Finish the analyzing process.
            self.finish_analyzing()

        t = threading.Thread(target=runner)
        t.start()

    # Finish analyzing and report the results.
    @staticmethod
    def finish_analyzing():
        # Enable the index and search buttons.
        enable_search_buttons()

        # Change the status.
        update_status("Finished searching")
        print()


# The part of the GUI which handles clustering settings and functions.
class KMeansClusteringFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing button to start clustering the data
        self.cluster_button = ttk.Button(self, text="Cluster", command=lambda: self.start_analyzing(),
                                         width=20, state="disabled")

        self.cluster_button.grid(row=1, column=6, sticky=E)

        # Advanced options for the querying.
        self.options_frame = Frame(self)
        self.options_frame.grid_columnconfigure(5, weight=1)
        self.options_frame.grid(row=1, column=1, columnspan=5, sticky=W + E, pady=(3, 0))

        self.weight_function = CompoundComboBox(self.options_frame,
                                                [scoring_mode for scoring_mode in KMclus.scoring_measures],
                                                "Weight Function: ")
        self.weight_function.grid(row=1, column=1, sticky=W, padx=(10, 0))

        self.stemmer = CompoundComboBox(self.options_frame,
                                        [stemmer for stemmer in KMclus.stemmer],
                                        "Stemmer: ")
        self.stemmer.grid(row=1, column=2, sticky=W)

        self.clusters_label = ttk.Label(self, text="Number of Clusters: ")
        self.clusters = ttk.Entry(self)

        self.clusters_label.grid(row=2, column=1, sticky=W)
        self.clusters.grid(row=2, column=2, sticky=W)

        self.runs_label = ttk.Label(self, text="Number of Runs: ")
        self.runs = ttk.Entry(self)

        self.runs_label.grid(row=2, column=3, sticky=W)
        self.runs.grid(row=2, column=4, sticky=W)

        # TODO: Secondly, add more options for both such as a choice for silhouette score and amount of

    # Start analyzing the query and find results.
    def start_analyzing(self):
        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the user input on the values to use
        weight_function = self.weight_function.get()
        stemmer = self.stemmer.get()
        # Check that all values are filled in
        if self.clusters.get() != '':
            if int(self.clusters.get()) > 0:
                clusters = int(self.clusters.get())
            else:
                print("Please fill in a value greater than 0 for clusters")
                enable_search_buttons()
                return
        else:
            print("Please fill in a value greater than 0 for clusters")
            enable_search_buttons()
            return
        if self.runs.get() != '':
            if int(self.runs.get()) > 0:
                runs = int(self.runs.get())
            else:
                print("Please fill in a value greater than 0 for runs")
                enable_search_buttons()
                return
        else:
            print("Please fill in a value greater than 0 for runs")
            enable_search_buttons()
            return

        # Change the status.
        update_status("Clustering... This may take a while.")
        print("=== CLUSTERING ===")
        print("Starting k-Means clustering with the following settings: ")
        print("- Weight function:", weight_function)
        print("- Stemmer:", stemmer)
        print("- Clusters:", clusters)
        print("- Runs:", runs)
        print()

        # Initialize the analyzer.
        def runner():
            # Calculate the scores.
            try:
                X, model = KMclus.clusterKMeans(stemmer, weight_function, clusters, runs)

                # When finished, pop up a plot frame.
                t = PlotFrame(gui, X, model, "KM", 0)
                t.wm_title("Window")

            except vsa.EmptyQueryException:  # TODO: What to do here
                print("Query is empty after normalization, please change the query.")

            # Finish the analyzing process.
            self.finish_analyzing()

        t = threading.Thread(target=runner)
        t.start()

    # Finish analyzing and re-enable the buttons.
    @staticmethod
    def finish_analyzing():
        # Enable the index and search buttons.
        enable_search_buttons()

        # Change the status.
        update_status("Finished clustering")
        print()

# The part of the GUI which handles clustering settings and functions.
class DBSCANClusteringFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing button to start clustering the data
        self.DBSCANcluster_button = ttk.Button(self, text="Cluster", command=lambda: self.start_analyzing(),
                                         width=20, state="disabled")

        self.DBSCANcluster_button.grid(row=1, column=6, sticky=E)

        # Advanced options for the querying.
        self.options_frame = Frame(self)
        self.options_frame.grid_columnconfigure(5, weight=1)
        self.options_frame.grid(row=1, column=1, columnspan=5, sticky=W + E, pady=(3, 0))

        self.weight_function = CompoundComboBox(self.options_frame,
                                                [scoring_mode for scoring_mode in DBclus.scoring_measures],
                                                "Weight Function: ")
        self.weight_function.grid(row=1, column=1, sticky=W, padx=(10, 0))

        self.stemmer = CompoundComboBox(self.options_frame,
                                        [stemmer for stemmer in DBclus.stemmer],
                                        "Stemmer: ")
        self.stemmer.grid(row=1, column=2, sticky=W)

        self.eps_label = ttk.Label(self, text="Eps: ")
        self.eps = ttk.Entry(self)

        self.eps_label.grid(row=2, column=1, sticky=W)
        self.eps.grid(row=2, column=2, sticky=W)

        self.min_samples_label = ttk.Label(self, text="min_samples: ")
        self.min_samples = ttk.Entry(self)

        self.min_samples_label.grid(row=2, column=3, sticky=W)
        self.min_samples.grid(row=2, column=4, sticky=W)

        # TODO: First get DBSCAN working

    # Start analyzing the query and find results.
    def start_analyzing(self):
        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Get the user input on the values to use
        weight_function = self.weight_function.get()
        stemmer = self.stemmer.get()
        # Check that all values are filled in
        if self.eps.get() != '':
            if float(self.eps.get()) > 0:
                eps = float(self.eps.get())
            else:
                print("Please fill in a value greater than 0 for eps")
                enable_search_buttons()
                return
        else:
            print("Please fill in a value greater than 0 for eps")
            enable_search_buttons()
            return
        if self.min_samples.get() != '':
            if int(self.min_samples.get()) > 0:
                min_samples = int(self.min_samples.get())
            else:
                print("Please fill in a value greater than 0 for min_samples")
                enable_search_buttons()
                return
        else:
            print("Please fill in a value greater than 0 for min_samples")
            enable_search_buttons()
            return

        # Change the status.
        update_status("Clustering... This may take a while.")
        print("=== CLUSTERING ===")
        print("Starting DBSCAN clustering with the following settings: ")
        print("- Weight function:", weight_function)
        print("- Stemmer:", stemmer)
        print("- Eps:", eps)
        print("- min_samples:", min_samples)
        print()

        # Initialize the analyzer.
        def runner():
            # Calculate the scores.
            try:
                X, model, n_clusters = DBclus.cluster(stemmer, weight_function, eps, min_samples)

                # When finished, pop up a plot frame.
                t = PlotFrame(gui, X, model, "KM", n_clusters)
                t.wm_title("Window")

            except vsa.EmptyQueryException:  # TODO: What to do here
                print("Query is empty after normalization, please change the query.")

            # Finish the analyzing process.
            self.finish_analyzing()

        t = threading.Thread(target=runner)
        t.start()

    # Finish analyzing and re-enable the buttons.
    @staticmethod
    def finish_analyzing():
        # Enable the index and search buttons.
        enable_search_buttons()

        # Change the status.
        update_status("Finished clustering")
        print()


class PlotFrame(Toplevel):
    def __init__(self, master, X, model, algorithm, n_clusters):
        super().__init__(master)

        # Draw the data we want!
        if algorithm == "KM":
            self.plot = KMclus.clutersgraph(X, model)
        else:
            self.plot = DBclus.clustergraph(X, model, n_clusters)

        # Instantiate canvas
        self.canvas = canvas = FigureCanvasTkAgg(self.plot, self)

        # Pack canvas into root window
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        # Instantiate and pack toolbar
        self.toolbar = toolbar = NavigationToolbar2TkAgg(canvas, self)

        # Show canvas and toolbar
        toolbar.update()
        canvas.show()


# The part of the GUI which handles classification.
class ClassificationFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(2, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Create a button to start indexing.
        self.classification_button = ttk.Button(self, text="Classification",
                                                command=lambda: self.start_classification(),
                                                width=20, state="disabled")
        self.classification_button.grid(row=0, column=3, sticky=E)

    def start_classification(self):
        # Disable the index and search buttons, as we don't want it to be pressed multiple times.
        disable_search_buttons()

        # Change the status.
        update_status("Classifying...")
        print("=== CLASSIFIER ===")
        print("Starting classification: ")
        print()

        # Initialize the classifier.
        def runner():
            classification.find_labels(indexer)
            classification.fit_data(indexer)
            classification.print_results()
            self.finish_classifying()

        t = threading.Thread(target=runner)
        t.start()

    @staticmethod
    def finish_classifying():
        # Enable the index and search buttons.
        enable_search_buttons()
        update_status("Finished classifying")


# The part of the GUI which views the results of the indexing and querying.
class ResultFrame(Frame):
    def __init__(self, master):
        super().__init__(master, bd=1, relief=SUNKEN)
        self.grid_columnconfigure(1, weight=1)
        self.pack(fill=BOTH, expand=1, padx=10, pady=10)

        self.result_frame_text = ScrolledText(self)
        # self.result_frame_text = ScrolledText(self, state="disabled")
        self.result_frame_text.pack(fill=BOTH, expand=1)

        # Redirect stdout and stderr to this frame.
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        sys.__stdout__.write(text)
        # self.result_frame_text.config(state="normal")
        self.result_frame_text.insert(END, text)
        # self.result_frame_text.config(state="disabled")
        self.result_frame_text.see(END)


# The status bar at the bottom of the frame.
class StatusFrame(Frame):
    def __init__(self, master):
        super().__init__(master, bd=1, relief=SUNKEN)
        self.grid_columnconfigure(1, weight=1)
        self.pack(fill=BOTH, expand=0)

        self.status_label = Label(self, text="Waiting", bd=1, relief=SUNKEN, anchor=W)
        self.status_label.pack(side=BOTTOM, fill=X)


# Separator lines between frames.
class SeparatorFrame(Frame):
    def __init__(self, master):
        super().__init__(master, height=2, bd=1, relief=SUNKEN)
        self.pack(fill=X, padx=5)


class InterfaceRoot(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(1, weight=1)
        self.pack(fill=BOTH, expand=1)

        # Add a notebook frame.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=BOTH, padx=10, pady=(10, 0))

        self.index_frame = IndexFrame(self)
        self.notebook.add(self.index_frame, text="Indexing")

        self.boolean_query_space = BooleanQueryFrame(self)
        self.notebook.add(self.boolean_query_space, text="Boolean queries")

        self.vector_space_query_frame = VectorSpaceQueryFrame(self)
        self.notebook.add(self.vector_space_query_frame, text="Vector space queries")

        self.probabilistic_query_frame = ProbabilisticQueryFrame(self)
        self.notebook.add(self.probabilistic_query_frame, text="Probabilistic queries")

        self.kmeansclustering_frame = KMeansClusteringFrame(self)
        self.notebook.add(self.kmeansclustering_frame, text="KMeansClustering")

        self.dbscansclustering_frame = DBSCANClusteringFrame(self)
        self.notebook.add(self.dbscansclustering_frame, text="DBSCANClustering")

        self.classification_frame = ClassificationFrame(self)
        self.notebook.add(self.classification_frame, text="Classification")

        self.result_frame = ResultFrame(self)
        self.status_frame = StatusFrame(self)


def update_status(status):
    gui.status_frame.status_label.config(text=status)


def print_results(query, results, use_crawler_data, has_score=True, top_x=10, score_format="%0.8f"):
    if len(results) == 0:
        print("No results found for query \"" + query + "\"!")
        return

    # Print general data, such as the query and number of results.
    print("query = \"" + query + "\"")
    print(min(len(results), top_x), "of", len(results), "results:")

    # The desired column lengths and column header.
    if has_score:
        column_lengths = {'Rank': len(str(top_x)) + 4, 'P.id': 4 + 2,
                          'Score': len(str(score_format % results[0][1])) + 2, 'Title': 80}
    else:
        column_lengths = {'P.id': 4 + 2, 'Title': 80}

    # Create the header of the table.
    header = "".join([header + " " * (length - len(header)) for header, length in column_lengths.items()])
    header = header.replace("Title", "Title + Crawler Data") if use_crawler_data else header

    print(header)
    print("-" * sum(column_lengths.values()))

    # Print all of the formatted results.
    for i in range(0, min(len(results), top_x)):
        if has_score:
            paper_id, score = results[i]
            paper = database.paper_id_to_paper[paper_id]
            print(i + 1, ".", " " * (column_lengths["Rank"] - len(str(i + 1)) - 1),
                  paper.id, " " * (column_lengths["P.id"] - len(str(paper.id))),
                  score_format % score, " " * (column_lengths["Score"] - len(str(score_format % score))),
                  paper.stored_title.lstrip(), sep="")
        else:
            paper = results[i]
            print(paper.id, " " * (column_lengths["P.id"] - len(str(paper.id))),
                  paper.stored_title, sep="")

        if use_crawler_data:
            print(get_info(paper.stored_title, indent_offset=sum(column_lengths.values())-column_lengths["Title"]))


def enable_search_buttons():
    gui.index_frame.indexing_button.config(state="normal")
    gui.classification_frame.classification_button.config(state="normal")
    gui.vector_space_query_frame.query_button.config(state="normal")
    gui.boolean_query_space.query_button.config(state="normal")
    gui.probabilistic_query_frame.query_button.config(state="normal")
    gui.kmeansclustering_frame.cluster_button.config(state="normal")
    gui.dbscansclustering_frame.DBSCANcluster_button.config(state="normal")


def disable_search_buttons():
    gui.index_frame.indexing_button.config(state="disabled")
    gui.classification_frame.classification_button.config(state="disabled")
    gui.vector_space_query_frame.query_button.config(state="disabled")
    gui.boolean_query_space.query_button.config(state="disabled")
    gui.probabilistic_query_frame.query_button.config(state="disabled")
    gui.kmeansclustering_frame.cluster_button.config(state="disabled")
    gui.dbscansclustering_frame.DBSCANcluster_button.config(state="disabled")


if __name__ == '__main__':
    root = Tk()
    root.title("[2IMM15] - Information retrieval module")
    root.minsize(width=1500, height=800)

    gui = InterfaceRoot(root)

    # Create the objects we will need.
    indexer = Indexer(update_status)
    analyzer = None

    # Start the main GUI loop.
    root.mainloop()
