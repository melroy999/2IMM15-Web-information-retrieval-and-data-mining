
class RuleFactory:

    # Superset of all rules, initialized to none
    all_rules = {
        'title': 'none',
        'author_iterator':  'none',
        'author': 'none',
        'doc_url': 'none',
        'isbn': 'none',
        'publisher': 'none',
        'publication_year': 'none',
        'publication': 'none'
    }

    # Commonly occuring rules
    common_rules = {
        'title': './div[contains(@class, "data")]/span[contains(@class, "title")]/text()',
        'author_iterator': './div[contains(@class, "data")]/span[contains(@itemprop, "author")]',
        'author': './*/span[contains(@itemprop, "name")]/text()',
        'doc_url': './nav/ul/li/div/a/@href',
        'publication_year': './div/span[contains(@itemprop, "datePublished")]/text()',
        'publication': './div/a/span/span[contains(@itemprop, "name")]/text()'
    }

    # Entry specific rules
    book_rules = {
        'title': common_rules['title'],
        'author_iterator': common_rules['author_iterator'],
        'author': common_rules['author'],
        'doc_url': common_rules['doc_url'],
        'isbn': './div/span[contains(@itemprop, "isbn")]/text()',
        'publisher': './div/span[contains(@itemprop, "publisher")]/text()',
        'publication_year': common_rules['publication_year'],
        'publication': common_rules['publication']
    }

    journal_rules = {
        'title': common_rules['title'],
        'author_iterator': common_rules['author_iterator'],
        'author': common_rules['author'],
        'doc_url': common_rules['doc_url'],
        'publication_year': common_rules['publication_year'],
        'publication': common_rules['publication']
    }

    reference_works_rules = {

    }

    collection_rules = {
        'title': common_rules['title'],
        'author_iterator': common_rules['author_iterator'],
        'author': common_rules['author'],
        'doc_url': common_rules['doc_url'],
        'publication_year': './div/a/span[contains(@itemprop, "datePublished")]/text()',
        'publication': common_rules['publication']
    }

    def get_rule_list(self, entry_type):
        if entry_type == 'entry book':
            return {**self.all_rules, **self.book_rules}
        elif entry_type == "entry reference":
            return {**self.all_rules, **self.reference_works_rules}
        elif entry_type == "entry article":
            return{**self.all_rules, **self.journal_rules}
        elif entry_type == "entry incollection":
            return {**self.all_rules, **self.collection_rules}
        else:
            # Bare minimum, if the entry is not any of the above, take only the title
            return {**self.all_rules, **{'title': self.common_rules['title']}}