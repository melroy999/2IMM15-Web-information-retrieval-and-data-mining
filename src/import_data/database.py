import sqlite3

# The location of the database.sqlite file, used when accessing the database.
_sqlite_file = "../../data/database.sqlite"

# The papers we have imported as an object.
papers = []

# A set of paper ids.
paper_ids = set()

# A mapping from paper id to paper.
paper_id_to_paper = {}

# The authors we have imported as an object.
authors = []

# A set of author ids.
author_ids = set()

# A mapping from author id to author.
author_id_to_author = {}


class Author:
    """
    Class for the authors in the database.
    """

    def __init__(self, author_data):
        self.id = author_data[0]
        self.name = author_data[1]
        self.papers = []


class PaperAuthors:
    """
    Class for the authors in the database.
    """

    def __init__(self, paper_author_data):
        self.id = paper_author_data[0]
        self.paper_id = paper_author_data[1]
        self.author_id = paper_author_data[2]


class Paper:
    """
    Class for the papers in the database.
    """

    def __init__(self, paper_data):
        self.id = paper_data[0]
        self.year = paper_data[1]
        self.title = paper_data[2]
        self.event_type = paper_data[3]
        self.pdf_name = paper_data[4]
        self.abstract = paper_data[5]
        self.paper_text = paper_data[6]
        self.authors = []


def _import_template(table, expression, where = ""):
    """
    Generic database table to object list conversion.

    @:type table: string
    @:param table: The name of the table to take the data from.

    @:type expression: lambda.
    @:param expression: lambda expression that converts a database object to a class object.

    @:rtype: list
    @:return: a list of objects as specified by the lambda expression.
    """

    # Variable that will contain the list of objects.
    object_list = []

    # Connect to the sqlite database, and select all in the specified table.
    with sqlite3.connect(_sqlite_file) as connection:
        c = connection.cursor()

        c.execute('SELECT * FROM "' + table + '" ' + where)
        for e in c.fetchall():
            paper = expression(e)
            object_list.append(paper)

    # Finally, return the list of objects.
    return object_list


def _import_authors():
    # Convert the authors table to a list of Author class objects.
    global authors
    global author_id_to_author
    global author_ids

    authors = _import_template('authors', lambda e: Author(e))
    author_id_to_author = {author.id: author for author in authors}
    author_ids = set(author_id_to_author.keys())

    return _import_template('authors', lambda e: Author(e))


def _import_papers():
    # Convert the papers table to a list of Paper class objects.
    global papers
    global paper_id_to_paper
    global paper_ids

    papers = _import_template('papers', lambda e: Paper(e), "WHERE NOT (id == 5820 OR id == 6178)")
    paper_id_to_paper = {paper.id: paper for paper in papers}
    paper_ids = set(paper_id_to_paper.keys())

    return papers


def import_data():
    # First import the papers and authors.
    _import_authors()
    _import_papers()

    # Now import the connection between authors and papers locally, as we don't need to store it.
    paper_to_author_table = _import_template('paper_authors', lambda e: PaperAuthors(e),
                                             "WHERE NOT (paper_id == 5820 OR paper_id == 6178)")

    # Now add the information to both the paper and author objects.
    for paper_author in paper_to_author_table:
        paper = paper_id_to_paper[paper_author.paper_id]
        author = author_id_to_author[paper_author.author_id]

        paper.authors.append(author.id)
        author.papers.append(paper.id)
