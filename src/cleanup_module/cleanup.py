import string
import unicodedata
import time
from collections import defaultdict

import pickle

import import_data.database as db

from information_retrieval.normalizer import Normalizer

"""
    Here we attempt to clean up the database in such a way that later operations will be more efficient.
    One defect in the original database file is that has a lot of noise in the data:
        - Unneeded control characters that will be counted as separate terms.
        - Combinations of long numbers and strings which have no meaning whatsoever.

    It is important to notice that only the paper_text seems to have this issue. Nevertheless, it shouldn't hurt
    applying the control character normalization to all text based sections of the papers.
"""

# The fields we may target for cleanup.
clean_up_paper_fields = ["title", "abstract", "paper_text"]

# A fast way of removing punctuation from a string, without replacement.
punctuation_removal_table = str.maketrans({key: "" for key in string.punctuation})

# A fast way of removing punctuation from a string, by replacing them with spaces.
punctuation_to_space = str.maketrans({key: " " for key in string.punctuation})

# Remove all punctuation, except for - and .
punctuation_to_space_with_exceptions = \
    str.maketrans({key: " " for key in string.punctuation.replace("-", "").replace(".", "")})

# Remove all numbers, and replace with spaces.
numbers_to_space = str.maketrans({key: " " for key in string.digits})

# A transition table to remove digits.
remove_digits = str.maketrans('', '', string.digits)

# List of vowels.
vowels = "aeiou"

# A list of homoglyphs that may occur in a word.
homoglyphs = {
    "o": ["0"],
    "0": ["o"],
    "i": ["l", "1"],
    "l": ["i", "1", "r"],
    "1": ["i", "l"],
    "f": ["t", "r", "l", "p"],
    "r": ["t", "f", "e"],
    "t": ["f", "r", "e", "h"],
    "v": ["w"],
    "w": ["v"],
    "m": ["n"],
    "n": ["m", "a", "u"],
    "rr": ["m"],
    "u": ["ti"],
    "y": ["w", "p"],
    "nn": ["rm"],
    "fl": ["ti"],
    "vr": ["wr"],
    "rrr": ["rm"],
    "q": ["n"],
    "s": ["c", "e"],
    "h": ["b"],
    "ll": ["n", "u"],
    "e": ["c", "a"],
    "fm": ["on"],
    "b": ["h"],
    "nun": ["mm"],
    "d": "c",
    "lc": ["k"]
}

# Cleanup, if it exists.
_cleanup = None


# We only want one instance...
def get_cleanup_instance(database=None):
    global _cleanup
    if _cleanup is None:
        _cleanup = _Cleanup(database)
    return _cleanup


class _Cleanup:
    def __init__(self, database):
        # Make sure that the database has imported the data.
        database.import_data()

        # Import the english word dictionary file.
        with open("../../data/word_list.txt") as file:
            words = file.readlines()
        self.word_list = set([word.strip().lower() for word in words])

        # Convert the author names to lower case tokens. Remove middle names.
        author_tokens_list = set()
        for author in database.authors:
            # Split the authors name.
            author_tokens = author.name.split()

            # For any of the tokens add the token if and only if it contains no punctuation.
            for token in author_tokens:
                if not any(p in token for p in string.punctuation):
                    author_tokens_list.add(token.lower())

        # Actually, add these authors to the list of words!
        self.word_list = self.word_list.union(author_tokens_list)

        # A scratch list for finding which elimination/selector defined whether the term is valid.
        self.scratch_list = defaultdict(int)
        self.timing_list = defaultdict(float)

        # Introduce a normalizer.
        self.normalizer = Normalizer("Nltk wordnet lemmatizer", False)
        self.original_terms = set()
        self.new_terms = set()
        self.original_terms_total_count = 0
        self.new_terms_total_count = 0

    # Remove terms in the text that are potentially noise.
    # This is the case for combinations of long numbers and long strings.
    def remove_potential_noise(self, text):
        # Split the original text on spaces, such that we can check for existing words.
        tokens = text.split()

        # Rebuild the string within this variable.
        output_tokens = []

        for i, token in enumerate(tokens):
            # Get the next word.
            next_word = ""
            if i < len(tokens) - 1:
                next_word = tokens[i + 1]

            self.original_terms.add(token)
            self.original_terms_total_count += 1

            # Check whether the token word exists as a real word, or is likely to be one.
            valid_word = self.validate_word_existence(token, next_word)
            if valid_word is not None:
                for term in valid_word.split():
                    self.new_terms.add(term)
                    self.new_terms_total_count += 1
                output_tokens.append(valid_word)

        # Rebuild the string.
        return " ".join(output_tokens)

    def update_performance_check(self, _start, value):
        self.timing_list[value] += time.time() - _start
        self.scratch_list[value] += 1

    def validate_word_existence(self, token, next_token):
        # In any case, we want to remove trailing punctuation.
        if token[-1] in ".,":
            token = token[:-1]

        # A version without any punctuation.
        no_punctuation_token = self.remove_punctuation(token, punctuation_removal_table)

        # A version without capital letters.
        lower_case_token = token.lower()

        # Get the lower case no punctuation version, as we want everything in lower case eventually.
        normalized_token = no_punctuation_token.lower().strip()

        # Empty characters are invalid.
        if len(normalized_token) == 0:
            return None

        # If there is a dash between the words, we have two separate words.
        _start = time.time()
        if "-" in token:
            self.update_performance_check(_start, "hyphen_approvals")
            return lower_case_token.replace("-", " ").replace(".", "")
        self.update_performance_check(_start, "hyphen_pass")

        # If it still contains dots, it is an abbreviation.
        _start = time.time()
        if "." in token:
            self.update_performance_check(_start, "abbreviation_approvals")
            return normalized_token
        self.update_performance_check(_start, "abbreviation_pass")

        # If the token is just one letter, we want to remove it, unless it is 'i' or 'a', which should pass immediately.
        _start = time.time()
        if len(normalized_token) == 1 and not normalized_token.isdecimal():
            if "ai".__contains__(normalized_token):
                self.update_performance_check(_start, "single_character_approvals")
                return normalized_token
            else:
                self.update_performance_check(_start, "single_character_eliminations")
                return None
        self.update_performance_check(_start, "single_character_pass")

        # If the word is in the world list, we can be sure that it is valid.
        _start = time.time()
        if normalized_token in self.word_list:
            self.update_performance_check(_start, "word_list_approvals")
            return normalized_token
        self.update_performance_check(_start, "word_list_pass")

        # We do not really care about numbers starting with 0.
        if normalized_token[0] == "0":
            return None

        # If the token is only digits it is probably a valid number.
        _start = time.time()
        if token.isdigit():
            # However, very long numbers are useless to us, so a limit of size 4 seems appropriate.
            # Next to that, we will not encounter meaningful numbers starting with 0.
            if len(token) <= 4:
                self.update_performance_check(_start, "is_digit_approvals")
                return normalized_token
            else:
                self.update_performance_check(_start, "is_invalid_digit_eliminations")
                return None
        self.update_performance_check(_start, "is_digit_pass")

        # A word without vowels could be a an abbreviation, but we filtered those out already.
        _start = time.time()
        if not any(c in vowels for c in normalized_token):
            self.update_performance_check(_start, "vowel_eliminations")
            return None
        self.update_performance_check(_start, "vowel_pass")

        # If the term is a combination of numbers and letters, we should see them as separate entities.
        _start = time.time()
        if any(c.isdigit() for c in normalized_token):
            words = self.remove_punctuation(normalized_token, numbers_to_space).split()
            numbers = normalized_token
            for word in words:
                numbers = numbers.replace(word, " ")
            numbers = [number for number in numbers.split() if len(token) <= 4]
            words = [word for word in words if word in self.word_list]

            self.update_performance_check(_start, "word_number_combination_alterations")
            return " ".join(words) + " " + " ".join(numbers)
        self.update_performance_check(_start, "word_number_combination_pass")

        # Certain words could be a name. They often start with a capital letter, with the rest lower case.
        # There is one exception to this rule, if it starts with Mc.
        _start = time.time()
        if no_punctuation_token.istitle() or no_punctuation_token.startswith("Mc") and \
                no_punctuation_token[2:].istitle():
            self.update_performance_check(_start, "name_structure_approvals")
            return normalized_token
        self.update_performance_check(_start, "name_structure_pass")

        # Some not found words seem to have homoglyph issues. Look at homoglyphs: f -> t, l -> 1, etc.
        # Here we replace all occurrences of the term. Later on we will replace them one by one.
        _start = time.time()
        for char, candidates in homoglyphs.items():
            for candidate in candidates:
                candidate_word = normalized_token.replace(char, candidate)
                if candidate_word in self.word_list:
                    self.update_performance_check(_start, "homoglyph_alterations")
                    return candidate_word
        self.update_performance_check(_start, "homoglyph_pass")

        # Again homoglyphs, but this time replacing occurrence by occurrence.
        for char, candidates in homoglyphs.items():
            for candidate in candidates:
                find = normalized_token.find(char)
                while find != -1:
                    # We encountered a match.
                    candidate_word = normalized_token[:find] + candidate + normalized_token[find + len(char):]
                    if candidate_word in self.word_list:
                        self.update_performance_check(_start, "homoglyph_extended_alterations")
                        return candidate_word
                    find = normalized_token.find(char, find + 1)
        self.update_performance_check(_start, "homoglyph_extended_pass")

        # Normalize, and check whether the normalized term occurs in the database.
        _start = time.time()
        normalized_normalized_token = self.normalizer.normalize(normalized_token)
        if normalized_normalized_token in self.word_list:
            self.update_performance_check(_start, "normalization_approvals")
            return normalized_token
        self.update_performance_check(_start, "normalization_pass")

        # A lot of words are split in half. Apply superglue.
        if normalized_token + self.remove_punctuation(next_token, punctuation_removal_table).lower() in self.word_list:
            self.update_performance_check(_start, "superglue_approvals")
            return normalized_token + self.remove_punctuation(next_token, punctuation_removal_table).lower()
        self.update_performance_check(_start, "superglue_pass")

        # All caps characters seem to be abbreviations as well.
        if all(c.isupper() for c in no_punctuation_token) or no_punctuation_token.endswith("s") \
                and all(c.isupper() for c in no_punctuation_token[:-1]):
            self.update_performance_check(_start, "all_caps_approval")
            return normalized_token[:-1] + normalized_token[-1].replace("s", "")
        self.update_performance_check(_start, "all_caps_pass")

        # Any word that has both upper case and lower case at this point is probably gibberish.
        if any(c.isupper() for c in no_punctuation_token) and any(c.islower() for c in no_punctuation_token):
            self.update_performance_check(_start, "mixed_casing_eliminations")
            return None
        self.update_performance_check(_start, "mixed_casing_pass")

        # Unfiltered tokens we can print.
        # print(token, " " * (30 - len(token)), no_punctuation_token, " " * (30 - len(no_punctuation_token)),
        #       normalized_token)

        # If we have not decided what to do with the word, keep it.
        self.scratch_list["undecided"] += 1
        return normalized_token

    # Remove control characters from the text.
    @staticmethod
    def remove_control_characters(text):
        result = ""
        for ch in text:
            if unicodedata.category(ch)[0] != "C":
                result += ch
            else:
                # We convert the control character to a space, so that we don't accidentally glue two words together.
                result += " "
        return result

    # Remove punctuation in the given text.
    @staticmethod
    def remove_punctuation(text, translator=punctuation_to_space):
        # Remove all punctuation.
        return text.translate(translator)

    # Clean the collection of papers. We alter the original list of papers here.
    def clean(self, _papers):
        # First, check if we have a pickle dump already for this.
        try:
            # Save the papers.
            _papers = self.load_pickle_dump("papers_with_cleanup")

            # Warning, the papers in the database will not have been changed! So re-calculate the pointers.
            db.papers = _papers
            db.recalculate_paper_pointers()
        except FileNotFoundError:
            # Print a warning that they should download the file, instead of waiting here for completion.
            print("Warning: cleanup can take up to 5 minutes.")

            print("Progress...")
            for i, paper in enumerate(_papers):
                if i % 650 == 0 and i > 0:
                    print(str(10 * i // 650) + "%")

                # We have multiple fields we want to clean up.
                for field in clean_up_paper_fields:
                    # Get the field value dynamically from the class by querying the attribute of the class.
                    field_value = paper.__getattribute__(field)

                    # We will replace control characters first, as it does not require tokenization to work.
                    field_value = _Cleanup.remove_control_characters(field_value)

                    # Remove all punctuation that will not influence the sentences, i.e. everything except for - and .
                    field_value = self.remove_punctuation(field_value, punctuation_to_space_with_exceptions)

                    # Now we want to remove potential noise from the value, which should reduce the amount of terms.
                    field_value = self.remove_potential_noise(field_value)

                    # Set the field in the paper copy.
                    paper.__setattr__(field, field_value)

            # Store the data we just found in a pickle file for faster access.
            self.create_pickle_dump("papers_with_cleanup", _papers)
        return _papers

    # Store the cleanup we found in a file, so that we can fetch it faster next time.
    @staticmethod
    def create_pickle_dump(filename, data):
        with open("../../data/" + filename + ".pickle", "wb") as output_file:
            pickle.dump(data, output_file)

    @staticmethod
    def load_pickle_dump(filename):
        with open("../../data/" + filename + ".pickle", "rb") as input_file:
            return pickle.load(input_file)

    def print_table(self):
        print("\\begin{table}[]")
        print("\\centering")
        print("\\caption{Preprocessing results table}")
        print("\\label{PreprocessingResultsTable}")
        print("\\begin{tabular}{l|lll}")
        print("rule & occurrences & total time & time per operation \\\\")
        print("\\hline")
        for term in sorted(self.scratch_list.keys()):
            print(term.replace("_", "\\_"), self.scratch_list[term], "%.5f" % self.timing_list[term],
                  "%.5e" % (self.timing_list[term] / self.scratch_list[term]), sep=" & ", end="\\\\ \n")

        print("\\end{tabular}")
        print("\\end{table}")


if __name__ == "__main__":
    start = time.time()
    _cleanup = _Cleanup(db)
    papers = _cleanup.clean(db.papers)

    print(_cleanup.scratch_list)
    print(_cleanup.timing_list)
    print("Original amount of terms:", _cleanup.original_terms_total_count)
    print("New amount of terms:", _cleanup.new_terms_total_count)
    print("Original amount of unique terms:", len(_cleanup.original_terms))
    print("New amount of unique terms:", len(_cleanup.new_terms))
    print()

    _cleanup.print_table()

    print()

    print("Total running time:", time.time() - start)
