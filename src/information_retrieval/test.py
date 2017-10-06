from information_retrieval.indexer import Indexer
from import_data import database

if __name__ == '__main__':
    indexer = Indexer()
    indexer.full_index("None", True, None)
