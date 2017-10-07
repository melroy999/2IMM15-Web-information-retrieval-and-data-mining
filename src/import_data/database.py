import sqlite3

# The location of the database.sqlite file, used when accessing the database.
_sqlite_file = "../../data/database.sqlite"

# The papers we have imported as an object.
papers = []

# A set of paper ids.
paper_ids = set()

# A mapping from paper id to paper.
paper_id_to_paper = {}
paper_id_to_list_id = {}

class Author:
    """
    Class for the authors in the database.
    """

    def __init__(self, author_data):
        self.id = author_data[0]
        self.name = author_data[1]


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


def _import_template(table, expression):
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

        c.execute('SELECT * FROM "' + table + '" WHERE NOT (id == 5820 OR id == 6178)')
        for e in c.fetchall():
            paper = expression(e)
            object_list.append(paper)

    # Finally, return the list of objects.
    return object_list


def import_authors():
    """
    Convert the authors table to a list of Author class objects.

    @:rtype: List of Author objects.
    @:return: The list of authors taken from the authors table in the database.
    """
    return _import_template('authors', lambda e: Author(e))


def import_papers():
    """
    Convert the papers table to a list of Paper class objects.

    @:rtype: List of Paper objects.
    @:return: The list of authors taken from the papers table in the database.
    """
    global papers
    global paper_id_to_paper
    global paper_id_to_list_id
    global paper_ids

    papers = _import_template('papers', lambda e: Paper(e))
    paper_id_to_paper = {paper.id: paper for paper in papers}
    paper_id_to_list_id = {paper.id: i for i, paper in enumerate(papers)}
    paper_ids = set(paper_id_to_paper.keys())

    return _import_template('papers', lambda e: Paper(e))
