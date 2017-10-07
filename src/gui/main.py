import threading
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from gui.util import create_tool_tip

# The part of the GUI which handles indexing settings and functions.
from information_retrieval.indexer import Indexer, paper_fields
from information_retrieval.normalizer import name_to_normalizer
from import_data import database
import information_retrieval.vector_space_analysis as vsa
import information_retrieval.boolean_analysis as ba

# Amount of results we can show.
results_to_show = [10, 20, 50, 100, 1000, 10000]


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

        self.stemming_field.config(width=25)
        self.stemming_label.grid(row=0, column=0, sticky=W)
        self.stemming_field.grid(row=0, column=1, sticky=W, padx=10)

        # Now add some checkboxes for other options that the indexer provides.
        self.find_stop_words = ttk.Checkbutton(self, text="Enable stopwords")
        self.find_stop_words.state(['!alternate'])
        self.find_stop_words.state(['selected'])
        self.find_stop_words.grid(row=0, column=2, sticky=W)

        # Create a button to start indexing.
        self.indexing_button = ttk.Button(self, text="Index papers", command=lambda: start_indexing(), width=20)
        self.indexing_button.grid(row=0, column=3, sticky=E)


def start_indexing():
    # Disable the index and search buttons, as we don't want it to be pressed multiple times.
    gui.index_frame.indexing_button.config(state="disabled")
    gui.vector_space_query_frame.query_button.config(state="disabled")

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
        indexer.full_index(stemming_mode, use_stopwords, update_status)
        finish_indexing()

    t = threading.Thread(target=runner)
    t.start()


def finish_indexing():
    # Enable the index and search buttons.
    gui.index_frame.indexing_button.config(state="normal")
    gui.vector_space_query_frame.query_button.config(state="normal")
    gui.boolean_query_space.query_button.config(state="normal")


class BooleanQueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = ttk.Label(self, text="Query: ")
        self.query_field = ttk.Entry(self)
        self.query_button = ttk.Button(self, text="Search", command=lambda: start_boolean_analyzing(), width=20,
                                       state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, columnspan=5, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=6, sticky=E)

        # Advanced options for the querying.
        self.target_field_var = StringVar(self)
        target_field_choices = [field for field in paper_fields]
        self.target_field_var.set(target_field_choices[-1])
        self.target_field_label = ttk.Label(self, text="Default target field: ")
        self.target_field_field = ttk.Combobox(self, values=target_field_choices, textvariable=self.target_field_var,
                                               state="readonly")

        self.target_field_field.config(width=25)
        self.target_field_label.grid(row=1, column=0, sticky=W)
        self.target_field_field.grid(row=1, column=1, sticky=W, padx=10)

        self.result_count_var = IntVar(self)
        result_count_choices = [results for results in results_to_show]
        self.result_count_var.set(result_count_choices[0])
        self.result_count_label = ttk.Label(self, text="#Results: ")
        self.result_count_field = ttk.Combobox(self, values=result_count_choices, textvariable=self.result_count_var,
                                               state="readonly")

        self.result_count_field.config(width=25)
        self.result_count_label.grid(row=1, column=2, sticky=W)
        self.result_count_field.grid(row=1, column=3, sticky=W, padx=10)


# Start analyzing the query and find results.
def start_boolean_analyzing():
    # Get the query.
    query = gui.boolean_query_space.query_field.get().strip()

    # Make sure we are not doing an empty query...
    if query == '':
        return

    # Disable the index and search buttons, as we don't want it to be pressed multiple times.
    gui.index_frame.indexing_button.config(state="disabled")
    gui.vector_space_query_frame.query_button.config(state="disabled")
    gui.boolean_query_space.query_button.config(state="disabled")

    # Get the query mode and whether we are searching for a document or not.
    target_default_field = gui.boolean_query_space.target_field_var.get()
    result_count = gui.boolean_query_space.result_count_var.get()

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
        results = ba.search(query, indexer, target_default_field)

        # Print the results.
        print_boolean_results(query, results, result_count)

        # Finish the analyzing process.
        finish_boolean_analyzing()

    t = threading.Thread(target=runner)
    t.start()


def print_boolean_results(query, results, top_x=10):
    if len(results) == 0:
        print("No results found for query \"" + query + "\"!")
        return

    print()
    print("=" * 70)
    print("query = \"" + query + "\"")
    print(min(len(results), top_x), "of", len(results), "results:")

    for i in range(0, min(len(results), top_x)):
        paper = results[i]
        print(str(i + 1) + ".\t", paper.id, "\t", paper.title)

    print("=" * 70)


# Finish analyzing and report the results.
def finish_boolean_analyzing():
    # Enable the index and search buttons.
    gui.index_frame.indexing_button.config(state="normal")
    gui.vector_space_query_frame.query_button.config(state="normal")
    gui.boolean_query_space.query_button.config(state="normal")

    # Change the status.
    update_status("Finished searching")
    print()


# The part of the GUI which handles query settings and functions.
class VectorSpaceQueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(5, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = ttk.Label(self, text="Query: ")
        self.query_field = ttk.Entry(self)
        self.query_button = ttk.Button(self, text="Search", command=lambda: start_vector_space_analyzing(), width=20,
                                       state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, columnspan=5, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=6, sticky=E)

        # Advanced options for the querying.
        self.search_method_var = StringVar(self)
        search_method_choices = [scoring_mode for scoring_mode in vsa.scoring_measures]
        self.search_method_var.set(search_method_choices[0])
        self.search_method_label = ttk.Label(self, text="Search method: ")
        self.search_method_field = ttk.Combobox(self, values=search_method_choices, textvariable=self.search_method_var,
                                                state="readonly")

        self.search_method_field.config(width=25)
        self.search_method_label.grid(row=1, column=0)
        self.search_method_field.grid(row=1, column=1, sticky=W, padx=10)

        self.target_field_var = StringVar(self)
        target_field_choices = [field for field in paper_fields]
        self.target_field_var.set(target_field_choices[-1])
        self.target_field_label = ttk.Label(self, text="Paper field: ")
        self.target_field_field = ttk.Combobox(self, values=target_field_choices, textvariable=self.target_field_var,
                                               state="readonly")

        self.target_field_field.config(width=25)
        self.target_field_label.grid(row=1, column=2, sticky=W)
        self.target_field_field.grid(row=1, column=3, sticky=W, padx=10)

        self.result_count_var = IntVar(self)
        result_count_choices = [results for results in results_to_show]
        self.result_count_var.set(result_count_choices[0])
        self.result_count_label = ttk.Label(self, text="#Results: ")
        self.result_count_field = ttk.Combobox(self, values=result_count_choices, textvariable=self.result_count_var,
                                               state="readonly")

        self.result_count_field.config(width=25)
        self.result_count_label.grid(row=1, column=4, sticky=W)
        self.result_count_field.grid(row=1, column=5, sticky=W, padx=10)

        self.find_comparable_papers = ttk.Checkbutton(self, text="Find similar papers")
        self.find_comparable_papers.state(['!alternate'])
        self.find_comparable_papers.grid(row=1, column=6, sticky=W)
        create_tool_tip(self.find_comparable_papers, "Enter the paper's title to find similar papers.")


# Start analyzing the query and find results.
def start_vector_space_analyzing():
    # Get the query.
    query = gui.vector_space_query_frame.query_field.get().strip()

    # Make sure we are not doing an empty query...
    if query == '':
        return

    # Disable the index and search buttons, as we don't want it to be pressed multiple times.
    gui.index_frame.indexing_button.config(state="disabled")
    gui.vector_space_query_frame.query_button.config(state="disabled")
    gui.boolean_query_space.query_button.config(state="disabled")

    # Get the query mode and whether we are searching for a document or not.
    find_similar_documents = gui.vector_space_query_frame.find_comparable_papers.instate(['selected'])
    query_score_mode = gui.vector_space_query_frame.search_method_var.get()
    target_field = gui.vector_space_query_frame.target_field_var.get()
    result_count = gui.vector_space_query_frame.result_count_var.get()

    # Change the status.
    update_status("Searching...")
    print("=== VECTOR-SPACE ANALYZER ===")
    print("Starting vector space search with the following settings: ")
    print("- Scoring mode:", query_score_mode)
    print("- Search for similar document:", find_similar_documents)
    print("- Target field:", target_field)
    print("- Number of results:", result_count)
    print()

    # Initialize the analyzer.
    def runner():
        # Calculate the scores.
        # query, indexer, field, scoring_measure="tf", similar_document_search=False
        scores = vsa.search(query, indexer, target_field, query_score_mode, find_similar_documents)

        if scores is not None:
            # Print the scores.
            print_vector_space_results(query, scores, result_count)

        # Finish the analyzing process.
        finish_vector_space_analyzing()

    t = threading.Thread(target=runner)
    t.start()


def print_vector_space_results(query, scores, top_x=10):
    if len(scores) == 0:
        print("No results found for query \"" + query + "\"!")
        return

    print()
    print("=" * 70)
    print("query = \"" + query + "\"")
    print(min(len(scores), top_x), "of", len(scores), "results:")

    for i in range(0, min(len(scores), top_x)):
        paper_id, score = scores[i]
        print(str(i + 1) + ".\t", paper_id, "\t", '%0.8f' % score, "\t", database.paper_id_to_paper[paper_id].title)

    print("=" * 70)


# Finish analyzing and report the results.
def finish_vector_space_analyzing():
    # Enable the index and search buttons.
    gui.index_frame.indexing_button.config(state="normal")
    gui.vector_space_query_frame.query_button.config(state="normal")
    gui.boolean_query_space.query_button.config(state="normal")

    # Change the status.
    update_status("Finished searching")
    print()


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


class IndexingInterface(Frame):
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

        self.result_frame = ResultFrame(self)
        self.status_frame = StatusFrame(self)


def update_status(status):
    gui.status_frame.status_label.config(text=status)


if __name__ == '__main__':
    root = Tk()
    root.title("[2IMM15] - Information retrieval module")
    root.minsize(width=1024, height=700)

    gui = IndexingInterface(root)

    # Create the objects we will need.
    indexer = Indexer()
    analyzer = None

    # Start the main GUI loop.
    root.mainloop()
