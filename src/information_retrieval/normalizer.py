# A table that will contain punctuation to be removed.
import string

import unicodedata

punctuation_removal_table = str.maketrans({key: None for key in string.punctuation})

# A transition table to remove digits.
remove_digits = str.maketrans('', '', string.digits)


class Normalizer:
    # Remove punctuation in the given text.
    @staticmethod
    def remove_punctuation(text):
        # Remove all punctuation.
        return text.translate(punctuation_removal_table)

    # Remove control characters from the text.
    @staticmethod
    def remove_control_characters(text):
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


# A list of english stop words.
english_stopwords = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and',
                     'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
                     'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has',
                     'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it',
                     'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my',
                     'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own',
                     'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the',
                     'their', 'them', 'then', 'there', 'these', 'they', 'this', 'to', 'too', 'us', 'wants', 'was', 'we',
                     'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would',
                     'yet', 'you', 'your'}
