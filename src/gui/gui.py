import threading
from tkinter import *
from tkinter.scrolledtext import ScrolledText

import information_retrieval.indexer as indexer

########################################################################################################################
# The tooltip class has been taken from http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml
########################################################################################################################
from information_retrieval.analyzer import analyzer_examples


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window"""

        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except TclError:
            pass
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create_tool_tip(widget, text):
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip(text)

    def leave(event):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

########################################################################################################################
# End of source annotation
########################################################################################################################

class IndexingInterface(Frame):

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        top_frame = Frame(master)
        top_frame.pack(fill=X, side=TOP)

        self.__init_indexing_frame(top_frame)

        separator_1 = Frame(top_frame, height=2, bd=1, relief=SUNKEN)
        separator_1.pack(fill=X, padx=5)

        self.__init_query_frame(top_frame)

        separator_2 = Frame(master, height=2, bd=1, relief=SUNKEN)
        separator_2.pack(fill=X, padx=5)

        self.__init_result_frame(master)

        self.__init_status_bar(master)

        sys.stdout = self
        sys.stderr = self

    def __init_indexing_frame(self, master):
        frame = Frame(master)
        frame.grid_columnconfigure(1, weight=1)
        frame.pack(fill=X, expand=1, padx=10, pady=10)

        # Now, make drop down menus for the type of stemming and the search method in a different frame.
        stemming_var = StringVar(frame)
        stemming_choices = ['stemming_1', 'stemming_2', 'stemming_3', 'stemming_4']
        stemming_var.set(stemming_choices[0])
        self.stemming_label = Label(frame, text="Stemmer: ")
        self.stemming_field = OptionMenu(frame, stemming_var, *stemming_choices)

        self.stemming_field.config(width=25)
        self.stemming_label.grid(row=0, column=0, sticky=W)
        self.stemming_field.grid(row=0, column=1, sticky=W)

        search_method_var = StringVar(frame)
        search_method_choices = ['tf', 'wf', 'tf.idf', 'wf.idf']
        search_method_var.set(search_method_choices[0])
        self.search_method_label = Label(frame, text="Search method: ")
        self.search_method_field = OptionMenu(frame, search_method_var, *search_method_choices)

        self.search_method_field.config(width=25)
        self.search_method_label.grid(row=1, column=0, sticky=W)
        self.search_method_field.grid(row=1, column=1, sticky=W)

        # Now add some checkboxes for other options that the indexer provides.
        self.find_stop_words_var = IntVar()
        self.find_stop_words_var.set(1)
        self.find_stop_words = Checkbutton(frame, text="Enable stopwords", variable=self.find_stop_words_var)
        self.find_stop_words.grid(row=0, column=3, sticky=W)

        self.find_comparable_papers = Checkbutton(frame, text="Search for similar paper")
        self.find_comparable_papers.grid(row=0, column=2, sticky=W)
        create_tool_tip(self.find_comparable_papers, "Enter the paper's title to find similar papers.")

        # Create a button to start indexing.
        self.indexing_button = Button(frame, text="Index papers", command=lambda: self.start_indexing(), width=20)
        self.indexing_button.grid(row=1, column=3, sticky=E+W)

    def __init_query_frame(self, master):
        frame = Frame(master)
        frame.grid_columnconfigure(1, weight=1)
        frame.pack(fill=X, expand=1, padx=10, pady=10)

        # Start by making a bar at the top containing a label, field and button for query search.
        self.query_label = Label(frame, text="Query: ")
        self.query_field = Entry(frame)
        self.query_button = Button(frame, text="Search", command=lambda: print("Searching"), width=20)

        self.query_label.grid(row=0, column=0, sticky=W)
        self.query_field.grid(row=0, column=1, sticky=E+W, padx=10)
        self.query_button.grid(row=0, column=2, sticky=E)

    def __init_result_frame(self, master):
        frame = Frame(master, bd=1, relief=SUNKEN)
        frame.grid_columnconfigure(1, weight=1)
        frame.pack(fill=BOTH, expand=1, padx=10, pady=10)

        self.result_frame_text = ScrolledText(frame, state="disabled")
        self.result_frame_text.pack(fill=BOTH, expand=1)

    def __init_status_bar(self, master):
        self.status_bar = Label(master, text="Waiting", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

    def write(self, text):
        self.result_frame_text.config(state="normal")
        self.result_frame_text.insert(END, text)
        self.result_frame_text.config(state="disabled")
        self.result_frame_text.see(END)

    def start_indexing(self):
        # Report on chosen settings.
        print("Indexing... please wait.")

        # Initialize the indexer.
        def runner():
            indexer.init()
            self.write("Completed indexing!")

            analyzer_examples()

        t = threading.Thread(target=runner)
        t.start()


if __name__ == '__main__':
    root = Tk()
    root.minsize(width=1024, height=700)

    gui = IndexingInterface(root)

    root.mainloop()






