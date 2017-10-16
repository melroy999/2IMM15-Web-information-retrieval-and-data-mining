import json
import string
import unicodedata
import time
from collections import defaultdict

import pickle

import import_data.database as db
import re

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

# A fast way of removing punctuation from a string.
punctuation_removal_table = str.maketrans({key: " " for key in string.punctuation})

# A transition table to remove digits.
remove_digits = str.maketrans('', '', string.digits)

# A list of homoglyphs that may occur in a word.
homoglyphs = {
    "o": ["0"],
    "0": ["o"],
    "i": ["l", "1"],
    "l": ["i", "1"],
    "1": ["i", "l"],
    "f": ["t, r"],
    "r": ["t, f"],
    "t": ["f, r"],
}

# List of vowels.
vowels = "aeiouy"

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

    # Remove terms in the text that are potentially noise.
    # This is the case for combinations of long numbers and long strings.
    def remove_potential_noise(self, text, sp):
        # Split the original text on spaces, such that we can check for existing words.
        tokens = text.split()

        # Rebuild the string within this variable.
        output_tokens = []

        for i, token in enumerate(tokens):
            if i > 0:
                prev_token = tokens[i - 1]
            else:
                prev_token = None

            try:
                next_token = tokens[i + 1]
            except IndexError:
                next_token = None

            # Check whether the token word exists as a real word, or is likely to be one.
            valid_word = self.validate_word_existence(prev_token, token, next_token, sp)
            if valid_word is not None:
                output_tokens.append(valid_word)

        # Rebuild the string.
        return " ".join(output_tokens)

    def validate_word_existence(self, prev_token, token, next_token, sp):
        # First get the lower case version, as we want everything in lower case eventually.
        lower_case_token = token.lower()

        # If the token is just one letter, we want to remove it, unless it is 'i' or 'a', which should pass immediately.
        if len(lower_case_token) == 1:
            if "ai".__contains__(lower_case_token):
                return lower_case_token
            else:
                return None

        # If the word is in the world list, we can be sure that it is valid.
        if self.word_list.__contains__(lower_case_token):
            self.scratch_list["word_list"] += 1
            return lower_case_token

        # If the token is only digits it is probably a valid number.
        if token.isdigit():
            # However, very long numbers are useless to us, so a limit of size 6 seems appropriate.
            # Next to that, we will not encounter meaningful numbers starting with 0.
            if token[0] != "0" and len(token) <= 6:
                self.scratch_list["is_digit"] += 1
                return lower_case_token
            else:
                self.scratch_list["is_invalid_digit_pruning"] += 1
                return None

        # If the word is upper case in its entirety, it is probably an abbreviation.
        if token.isupper():
            self.scratch_list["is_upper"] += 1
            return lower_case_token

        # If the word is upper cased followed by an "s", it is probably an abbreviation as well.
        # However, we do not want the [s] ending in there... or do we? We don't eliminate it during search...
        # With this reasoning we always have to do both cases. So remove the s.
        if token[:-1].isupper() and token[-1] == "s":
            self.scratch_list["is_upper_plural"] += 1
            return lower_case_token[:-1]

        # If the word starts with a capital letter and is lower case on all the other letters, it is probably a name.
        if token.istitle():
            self.scratch_list["is_title"] += 1
            return lower_case_token

        # Does the word start with Mc and then a camel case? Probably a name as well...
        if token.startswith("Mc") and token[2:].istitle():
            self.scratch_list["is_mc_title"] += 1
            return lower_case_token

        # At this point, any combination of number and digit is probably useless.
        if any(c.isdigit() for c in lower_case_token):
            self.scratch_list["is_digit_number_pruning"] += 1
            return None

        # Words ending on 'th' indicate a ranking, which probably is not useful.
        if lower_case_token.endswith("th"):
            self.scratch_list["is_th_pruning"] += 1
            return None

        # Two term words are probably not useful.
        # Most four term words look like gibberish... naturally some would be useful, but most are useless.
        if len(lower_case_token) < 5:
            self.scratch_list["is_too_short_pruning"] += 1
            return None

        # On the other hand, long words don't look like gibberish.
        if len(lower_case_token) > 12:
            self.scratch_list["is_long_validity"] += 1
            return lower_case_token

        # Some papers seem to use a terrible version of english german. Try to replace all v with w.
        w_lower_case_token = lower_case_token.replace("v", "w")
        if self.word_list.__contains__(w_lower_case_token):
            self.scratch_list["is_bad_german_vord_altering"] += 1
            return w_lower_case_token

        # A lot of papers seem to have random spaces everywhere between certain words...
        # Recen tly, dissimilari ty, sampl es, su bstitu te, compu ters, etc
        if prev_token is not None and next_token is not None:
            if self.word_list.__contains__(prev_token + lower_case_token + next_token):
                self.scratch_list["superglue_both_sides"] += 1
                return prev_token + lower_case_token + next_token
            if self.word_list.__contains__(lower_case_token + next_token):
                self.scratch_list["superglue_right_side"] += 1
                return lower_case_token + next_token
            if self.word_list.__contains__(prev_token + lower_case_token):
                self.scratch_list["superglue_left_side"] += 1
                return prev_token + lower_case_token

        # Some not found words seem to have homoglyph issues. Look at homoglyphs: f -> t, l -> 1, etc.
        for char, candidates in homoglyphs.items():
            for candidate in candidates:
                candidate_word = lower_case_token.replace(char, candidate)
                if self.word_list.__contains__(candidate_word):
                    self.scratch_list["homoglyphs"] += 1
                    return candidate_word

        # At this point, any word with mixed capitalization is probably useless.
        if any(char.isupper() for char in token):
            self.scratch_list["capitalization_elimination_pruning"] += 1
            return None

        # Words without vowels are useless.
        if not any(char in vowels for char in token):
            self.scratch_list["no_vowels_pruning"] += 1
            return None

        # Use a spelling corrector and see if the word exists afterwards.
        spelling_correction = sp.correction(lower_case_token)
        if self.word_list.__contains__(spelling_correction):
            self.scratch_list["spelling_correction"] += 1
            return spelling_correction

        # Other ideas:
        # Use hyphenation to see if the word is pronounceable.
        # Split words on term combinations which usually would be spaced.
        # Use the stemmers and lemmatizers already here to see if the word exists.

        # Unfiltered tokens we can print.
        # print(token)
        # if spelling_correction != lower_case_token:
        #     print("Spelling corrected but invalid:", token, spelling_correction)
        # If we have not decided what to do with the word, keep it.

        self.scratch_list["undecided"] += 1
        return lower_case_token

    # Remove a prefix from a term.
    @staticmethod
    def remove_prefix(term, prefix):
        return term[len(prefix):]

    # Remove a postfix from a term.
    @staticmethod
    def remove_postfix(term, prefix):
        return term[:-len(prefix)]

    # Convert a camel case string to components.
    @staticmethod
    def camel_case_split(token):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
        return [m.group(0) for m in matches]

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
    def remove_punctuation(text):
        # Remove all punctuation.
        return text.translate(punctuation_removal_table)

    # Clean the collection of papers. We alter the original list of papers here.
    def clean(self, papers):
        # First, check if we have a pickle dump already for this.
        try:
            # Save the papers.
            papers = self.load_pickle_dump("papers_with_cleanup")

            # Warning, the papers in the database will not have been changed! So re-calculate the pointers.
            db.papers = papers
            db.recalculate_paper_pointers()
        except FileNotFoundError:
            # Print a warning that they should download the file, instead of waiting here for completion.
            print("Warning: cleanup can take up to 20 minutes. Please download the 'papers_with_cleanup.pickle' file "
                  "found on google drive.")

            # We only want to import the spelling checker when it is needed.
            import cleanup_module.spelling_checker as sp

            for paper in papers:
                print(paper.id)

                # We have multiple fields we want to clean up.
                for field in clean_up_paper_fields:
                    # Get the field value dynamically from the class by querying the attribute of the class.
                    field_value = paper.__getattribute__(field)

                    # We will replace control characters first, as it does not require tokenization to work.
                    field_value = _Cleanup.remove_control_characters(field_value)

                    # Now make sure that the punctuation is in a state such that split works well.
                    # I.e. replaced by spaces.
                    field_value = _Cleanup.remove_punctuation(field_value)

                    # Now we want to remove potential noise from the value, which should reduce the amount of terms.
                    field_value = self.remove_potential_noise(field_value, sp)

                    # Set the field in the paper copy.
                    paper.__setattr__(field, field_value)

            # Store the data we just found in a pickle file for faster access.
            self.create_pickle_dump("papers_with_cleanup", papers)
        return papers

    # Store the cleanup we found in a file, so that we can fetch it faster next time.
    @staticmethod
    def create_pickle_dump(filename, data):
        with open("../../data/" + filename + ".pickle", "wb") as output_file:
            pickle.dump(data, output_file)

    @staticmethod
    def load_pickle_dump(filename):
        with open("../../data/" + filename + ".pickle", "rb") as input_file:
            return pickle.load(input_file)


if __name__ == "__main__":
    start = time.time()
    _cleanup = _Cleanup(db)
    papers = _cleanup.clean(db.papers)
    for paper in papers[:10]:
        print(paper.paper_text)
    print()
    for paper in db.papers[:10]:
        print(paper.paper_text)

    print(_cleanup.scratch_list)
    print(time.time() - start)
