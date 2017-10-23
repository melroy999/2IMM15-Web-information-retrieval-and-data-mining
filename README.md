# 2IMM15-Web-information-retrieval-and-data-mining

## Instructions for the GUI controlled components
Components: preprocessor, indexer, query system, clustering and classification

Please download the following files:
- database.sqlite: https://melroy.pro/uploads/database.sqlite
- crawler_sqlite.db: https://melroy.pro/uploads/crawler_sqlite.db

The components above have been merged into the master branch. Please take the following steps to get this component working:
1. Clone the master branch of the git repository into a directory of choice.
2. Import the directory in PyCharm. Once imported, select the “src” folder, right click and choose the option “Mark Directory as > Source Root”
3. Before proceeding, please insert the crawler_sqlite.db and database.sqlite files into the data folder. Here the database.sqlite is the database provided by the NIPS collection, and crawler_sqlite.db is a file provided by us. 
4. Check all python files for missing packages. 
5. Run the setup.py script to initialize the WordNet stemmer database.
6. Run main.py in the gui folder to start the GUI. 
7. Start by clicking on the “Index papers” button. This will automatically call the preprocessing. 
8. Once indexing is done, the components can be accessed and interacted with through their corresponding tabs. 
