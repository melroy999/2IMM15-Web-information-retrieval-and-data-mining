import threading
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from TopicModeling import TM
from TopicModeling import LDA_TM
from TopicModeling import A_TM

# Amount of results we can show.
results_to_show = [10, 20, 50, 100, 1000, 10000]


class TopicModelingFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(2, weight=1)
        self.pack(fill=X, expand=0, padx=10, pady=10)


        # Create a button to start creating or loading existing data.
        self.n_topic_label = ttk.Label(self, text="Number of Topics: ")
        self.n_topic_field = ttk.Entry(self)
        self.author_name_label = ttk.Label(self, text="Author Name: ")
        self.author_name_field = ttk.Entry(self)
        self.paper_id_label=ttk.Label(self, text="Document Id: ")
        self.paper_id_field=ttk.Entry(self)
        self.text_query_label=ttk.Label(self, text="Query: ")
        self.text_query_field=ttk.Entry(self)
        self.year_label=ttk.Label(self, text="Year: ")
        self.year_field=ttk.Entry(self)

        self.topic_num_label=ttk.Label(self, text="Topic Number: ")
        self.topic_num_field=ttk.Entry(self)



        self.load_data = ttk.Button(self, text="Load existing Model",
                                                command=lambda: self.LoadModel(),
                                                width=20)
        self.create_data = ttk.Button(self, text="Create Model",
                                                command=lambda: self.CreateModel(),
                                                width=20)

        self.get_author_info = ttk.Button(self, text="Get Author Info",
                                        command=lambda: self.Tm.get_author_info(self.author_name_field.get().strip()),
                                        width=20)

        self.get_document_info = ttk.Button(self, text="Get Document Info",
                                        command=lambda: self.Tm.get_doc_info(self.paper_id_field.get().strip()),
                                        width=20)

        self.get_query_info = ttk.Button(self, text="Get Query Results",
                                        command=lambda: self.GetQueryResults(),
                                        width=20)

        self.TopicIdevolution = ttk.Button(self, text="Topic(Id) Evolution",
                                        command=lambda: self.Tm.lda.topic_evolution_by_year(int(self.topic_num_field.get().strip())),
                                        width=20)
        self.allTopicsEvolution = ttk.Button(self, text="PlotAllTopicsEvolution",
                                        command=lambda: self.Tm.lda.plot_all_topic_evolutions(),
                                        width=20)
        self.print_top_topics_of_year = ttk.Button(self, text="Print Top Topics of Year",
                                command=lambda: self.Tm.lda.print_top_topics_of_year(int(self.year_field.get().strip())),
                                width=20)
        self.plot_author_cluster = ttk.Button(self, text="Author Clustering",
                                command=lambda: self.Tm.atm.plot_author_tsne_plot(),
                                width=20)

        self.printAuthorTopics = ttk.Button(self, text="Print Author Topics",
                                command=lambda: self.Tm.atm.printtopics(),
                                width=20)

        self.print_top_titles_by_topic = ttk.Button(self, text="Top Titles by Topic",
                                command=lambda: self.Tm.lda.print_top_titles_by_topic(),
                                width=20)

        self.plot_doc_cluster = ttk.Button(self, text="Document Clustering",
                                command=lambda: self.Tm.lda.plot_evolution_plots(),
                                width=20)

        self.plot_author_cluster = ttk.Button(self, text="Author Clustering",
                                command=lambda: self.Tm.atm.plot_author_tsne_plot(),
                                width=20)


        self.plot_doc_cluster_interia = ttk.Button(self, text="Document Clustering Evaluation",
                                command=lambda: self.Tm.lda.plot_doc_clustering_interia(max_cluster=20),
                                width=30)


        self.plot_author_cluster_interia = ttk.Button(self, text="Author Clustering Evaluation",
                                command=lambda: self.Tm.atm.plot_author_clustering_interia(max_cluster=20),
                                width=30)

        self.plot_doc_classification = ttk.Button(self, text="Document Classification",
                                command=lambda: self.Tm.lda.create_classification_from_cluster_data(),
                                width=20)


        self.plot_author_classification = ttk.Button(self, text="Author Classification",
                                command=lambda: self.Tm.atm.create_classification_from_cluster_data(),
                                width=20)

















        self.n_topic_label.grid(row=0, column=0, sticky=W)
        self.n_topic_field.grid(row=0, column=1,  sticky= W)
        self.load_data.grid(row=0, column=2, sticky=W)
        self.create_data.grid(row=0, column=3, sticky=W)


        self.author_name_label.grid(row=0, column=4, sticky=W)
        self.author_name_field.grid(row=0, column=5, sticky= W)
        self.get_author_info.grid(row=0, column=6, sticky=W)


        self.text_query_label.grid(row=0, column=7, sticky=W)
        self.text_query_field.grid(row=0, column=8,sticky= W)
        self.get_query_info.grid(row=0, column=9, sticky=W)


        self.paper_id_label.grid(row=1, column=6, sticky=W)
        self.paper_id_field.grid(row=1, column=7,  sticky= W)
        self.get_document_info.grid(row=1, column=8, sticky=W)

        self.year_label.grid(row=1, column=0, sticky=W)
        self.year_field.grid(row=1, column=1,sticky= W)


        self.topic_num_label.grid(row=1, column=2, sticky=W)
        self.topic_num_field.grid(row=1, column=3,sticky= W)
        self.TopicIdevolution.grid(row=1, column=4, sticky=W)
        self.print_top_titles_by_topic.grid(row=1, column=5, sticky=W)

        self.get_query_info.grid(row=0, column=10, sticky=W)
        self.allTopicsEvolution.grid(row=2, column=1, sticky=W)

        self.print_top_topics_of_year.grid(row=2, column=2, sticky=W)
        self.plot_author_cluster.grid(row=2, column=3, sticky=W)

        self.printAuthorTopics.grid(row=2, column=4, sticky=W)
        self.plot_doc_cluster.grid(row=2, column=5, sticky=W)



        self.plot_doc_cluster_interia.grid(row=2, column=6, sticky=W)
        self.plot_author_cluster_interia.grid(row=2, column=7, sticky=W)
        #self.plot_doc_classification.grid(row=2, column=7, sticky=W)
        #self.plot_author_classification.grid(row=2, column=6, sticky=W)



















    def LoadModel(self):
        n_topics = self.n_topic_field.get().strip()
        if n_topics == '':
            return

        n_topics = int(n_topics)
        self.Tm=TM.TM(n_topics)
        self.Tm.load_existing_model()
        return

    def CreateModel(self):
        n_topics = self.n_topic_field.get().strip()
        if n_topics == '':
            return
        try:
            n_topics = int(n_topics)
            self.Tm=TM.TM(n_topics)
            self.Tm.create_model()
        except ValueError:
            print("Please enter a valid number for the number of topics field.")
        return

    def GetQueryResults(self):
        print(self.Tm.lda.ldaquery(self.text_query_field.get().strip()))


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

        self.topicmodeling_frame = TopicModelingFrame(self)
        self.notebook.add(self.topicmodeling_frame, text="TopicModeling")

        self.result_frame = ResultFrame(self)
        self.status_frame = StatusFrame(self)


def update_status(status):
    gui.status_frame.status_label.config(text=status)


if __name__ == '__main__':
    root = Tk()
    root.title("[2IMM15] - Information retrieval module - topic modeling")
    root.minsize(width=1500, height=800)

    gui = InterfaceRoot(root)

    # Start the main GUI loop.
    root.mainloop()
