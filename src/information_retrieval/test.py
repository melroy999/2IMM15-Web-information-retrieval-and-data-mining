import pickle
from collections import Counter, defaultdict

import numpy
import pandas
import time

import import_data.database as database

# We want to create a 2d array, where rows are terms and columns are tf, wf etc
from information_retrieval.normalizer import Normalizer

# Fields
fields = ["paper_text", "abstract", "title"]


def create_table_file(filename, data):
    with open(filename + ".pickle", "wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_table_file(filename):
    with open(filename + ".pickle", "rb") as input_file:
        return pickle.load(input_file)


import unicodedata


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


mpa = str.maketrans(dict.fromkeys(range(32)))


def calc_paper_data(paper, global_counters, normalizer):
    # The dictionary generated by this paper.
    result = {}

    # Iterate over all fields.
    for field in fields:
        # Initialize the field dictionary to be a dictionary.
        result[field] = {}

        # Load the data we want to analyze.
        text = paper.__getattribute__(field).translate(mpa)

        # Convert to lower case, remove punctuation and tokenize.
        tokens = normalizer.remove_punctuation(text.lower()).split()
        result[field]["number_of_terms"] = len(tokens)

        # Count the term frequency, and also update the global counter.
        term_frequencies = defaultdict(int)
        for term, value in Counter(tokens).items():
            if normalizer.is_valid_term(term):
                norm_term = normalizer.normalize(term)
                term_frequencies[norm_term] = term_frequencies[norm_term] + value
        global_counters[field].update(term_frequencies)
        result[field]["number_of_unique_terms"] = len(term_frequencies)

        # Create the table with the data.
        table_contents = {"tf": term_frequencies}
        frequency_data = pandas.DataFrame(table_contents)

        # Calculate additional data, like the weighted term frequency.
        result[field]["frequencies"] = frequency_data.assign(wf=numpy.log10(frequency_data["tf"]) + 1)

        # Calculate the vector lengths.
        result[field]["vector_lengths"] = {
            "tf": numpy.math.sqrt(sum(result[field]["frequencies"]["tf"] ** 2)),
            "wf": numpy.math.sqrt(sum(result[field]["frequencies"]["wf"] ** 2))
        }

    # Return the data.
    return result


# noinspection PyUnresolvedReferences
def calc_data(normalizer):
    # First get data in to the database.
    database.import_papers()

    # Start with importing all the term frequency data required.
    data = {"N": len(database.papers)}
    global_counters = {field: Counter() for field in fields}

    # Iterate over all the papers, and gather all the data.
    data["papers"] = {}
    for paper in database.papers:
        data["papers"][paper.id] = calc_paper_data(paper, global_counters, normalizer)

    # Save the collection data.
    data["collection"] = {}
    for field in fields:
        field_data = {}

        # Before noting down the collection frequency, we will first have to calculate the document frequency.
        document_frequency = defaultdict(int)
        for paper in database.papers:
            for term in data["papers"][paper.id][field]["frequencies"]["tf"]:
                document_frequency[term] += 1

        # Create the tables.
        # table_contents = {"cf": global_counters[field], "df": document_frequency}
        # frequency_data = pandas.DataFrame(table_contents)

        # Calculate idf.
        # data["collection"][field]["frequencies"] = frequency_data.assign(
        #     idf=numpy.log10(data["N"] / frequency_data["df"])
        # )

        table_contents = {"cf": global_counters[field], "df": global_counters[field]}
        field_data["frequencies"] = pandas.DataFrame(table_contents)

        # Get total number information.
        field_data["number_of_terms"] = field_data["frequencies"]["cf"].sum()
        field_data["number_of_unique_terms"] = len(global_counters[field])

        # Calculate the vector lengths.
        # data["collection"][field]["vector_lengths"] = {
        #     "cf": numpy.math.sqrt(sum(data["collection"][field]["frequencies"]["cf"] ** 2)),
        #     "df": numpy.math.sqrt(sum(data["collection"][field]["frequencies"]["df"] ** 2)),
        #     "idf": numpy.math.sqrt(sum(data["collection"][field]["frequencies"]["idf"] ** 2)),
        # }

        field_data["vector_lengths"] = {
            "cf": numpy.math.sqrt(sum(field_data["frequencies"]["cf"] ** 2))
        }

        # Add all data to the data dictionary.
        data["collection"][field] = field_data

    # dump the file.
    create_table_file("../../data/calc_" + normalizer.operator_name.lower().replace(" ", "_"), data)

    # Return the data.
    return data


def load_index():
    # Get the normalizer we want to use.
    normalizer = Normalizer(True, "None")

    # Check if the file exists, if not recalculate.
    try:
        data = load_table_file("../../data/calc_" + normalizer.operator_name.lower().replace(" ", "_"))
    except (FileNotFoundError, EOFError):
        print("Pre-calculated data file not found. Recalculating... this may take a long time.")
        data = calc_data(normalizer)

    return data


start = time.time()
result = load_index()
print('result["N"]', result["N"])
print('result["collection"]["paper_text"]["number_of_unique_terms"]',
      result["collection"]["paper_text"]["number_of_unique_terms"])
print('result["collection"]["paper_text"]["number_of_terms"]', result["collection"]["paper_text"]["number_of_terms"])
print('result["collection"]["paper_text"]["frequencies"]')
print(result["collection"]["paper_text"]["frequencies"])
print('result["collection"]["paper_text"]["frequencies"]["cf"]["neural"]')
print(result["collection"]["paper_text"]["frequencies"]["cf"]["neural"])
print('result["collection"]["paper_text"]["vector_lengths"]', result["collection"]["paper_text"]["vector_lengths"])
print()
print("time: ", time.time() - start)
