import threading
from tkinter import *
from tkinter.scrolledtext import ScrolledText

import information_retrieval as indexer

from gui.util import create_tool_tip
from information_retrieval.analyzer import Analyzer

# The part of the GUI which handles indexing settings and functions.
from information_retrieval.indexer import Indexer


class IndexFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(1, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Now, make drop down menus for the type of stemming and the search method in a different frame.
        stemming_var = StringVar(self)
        stemming_choices = ['stemming_1', 'stemming_2', 'stemming_3', 'stemming_4']
        stemming_var.set(stemming_choices[0])
        self.stemming_label = Label(self, text="Stemmer: ")
        self.stemming_field = OptionMenu(self, stemming_var, *stemming_choices)

        self.stemming_field.config(width=25)
        self.stemming_label.grid(row=0, column=0, sticky=W)
        self.stemming_field.grid(row=0, column=1, sticky=W)

        search_method_var = StringVar(self)
        search_method_choices = ['tf', 'wf', 'tf.idf', 'wf.idf']
        search_method_var.set(search_method_choices[0])
        self.search_method_label = Label(self, text="Search method: ")
        self.search_method_field = OptionMenu(self, search_method_var, *search_method_choices)

        self.search_method_field.config(width=25)
        self.search_method_label.grid(row=1, column=0, sticky=W)
        self.search_method_field.grid(row=1, column=1, sticky=W)

        # Now add some checkboxes for other options that the indexer provides.
        self.find_stop_words_var = IntVar()
        self.find_stop_words_var.set(1)
        self.find_stop_words = Checkbutton(self, text="Enable stopwords", variable=self.find_stop_words_var)
        self.find_stop_words.grid(row=0, column=3, sticky=W)

        self.find_comparable_papers = Checkbutton(self, text="Search for similar paper")
        self.find_comparable_papers.grid(row=0, column=2, sticky=W)
        create_tool_tip(self.find_comparable_papers, "Enter the paper's title to find similar papers.")

        # Create a button to start indexing.
        self.indexing_button = Button(self, text="Index papers", command=lambda: start_indexing(), width=20)
        self.indexing_button.grid(row=1, column=3, sticky=E + W)


def start_indexing():
    # Disable the index and search buttons, as we don't want it to be pressed multiple times.
    gui.index_frame.indexing_button.config(state="disabled")
    gui.query_frame.query_button.config(state="disabled")

    # Change the status.
    update_status("Indexing...")
    print("Starting indexing with the following settings: ")
    print(" - Whoops none yet...")

    # Initialize the indexer.
    def runner():
        indexer.index()
        finish_indexing()

    t = threading.Thread(target=runner)
    t.start()


def finish_indexing():
    # Enable the index and search buttons.
    gui.index_frame.indexing_button.config(state="normal")
    gui.query_frame.query_button.config(state="normal")

    # Change the status.
    update_status("Finished indexing")
    print("Finished indexing")


# The part of the GUI which handles query settings and functions.
class QueryFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(1, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        # Initially disabled button, as we need to index first.
        self.query_label = Label(self, text="Query: ")
        self.query_field = Entry(self)
        self.query_button = Button(self, text="Search", command=lambda: start_analyzing(), width=20, state="disabled")

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, sticky=E + W, padx=10)
        self.query_button.grid(row=0, column=2, sticky=E)


def start_analyzing():
    # Disable the index and search buttons, as we don't want it to be pressed multiple times.
    gui.index_frame.indexing_button.config(state="disabled")
    gui.query_frame.query_button.config(state="disabled")

    # Change the status.
    update_status("Searching...")
    print("Starting searching with the following settings: ")
    print(" - Whoops none yet...")

    # Update the analyzer.
    global analyzer
    analyzer = Analyzer(indexer)

    # Initialize the analyzer.
    def runner():
        analyzer.analyzer_examples()
        finish_analyzing()

    t = threading.Thread(target=runner)
    t.start()


def finish_analyzing():
    # Enable the index and search buttons.
    gui.index_frame.indexing_button.config(state="normal")
    gui.query_frame.query_button.config(state="normal")

    # Change the status.
    update_status("Finished searching")
    print("Finished searching")


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

        self.index_frame = IndexFrame(self)

        self.separator_1 = SeparatorFrame(self)

        self.query_frame = QueryFrame(self)

        self.separator_2 = SeparatorFrame(self)

        self.result_frame = ResultFrame(self)

        self.status_frame = StatusFrame(self)


def update_status(status):
    gui.status_frame.status_label.config(text=status)


if __name__ == '__main__':
    root = Tk()
    root.minsize(width=1024, height=700)

    gui = IndexingInterface(root)

    # Create the objects we will need.
    indexer = Indexer()
    analyzer = Analyzer(indexer)

    # Start the main GUI loop.
    root.mainloop()








