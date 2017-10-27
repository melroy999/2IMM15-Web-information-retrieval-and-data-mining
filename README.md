# 2IMM15-Web-information-retrieval-and-data-mining

## Instructions for the GUI controlled components
Components: preprocessor, indexer, query system, clustering and classification

The database files can be downloaded through the following link:
https://melroy.pro/uploads/web_database_files.zip

The components above have been merged into the master branch. Please take the following steps to get this component working:
1. Clone the master branch of the git repository into a directory of choice.
2. Import the directory in PyCharm. Once imported, select the “src” folder, right click and choose the option “Mark Directory as > Source Root”
3. Before proceeding, please insert the crawler_sqlite.db and database.sqlite files into the data folder. Here the database.sqlite is the database provided by the NIPS collection, and crawler_sqlite.db is a file provided by us.
4. Check all python files for missing packages.
5. Run the setup.py script to initialize the WordNet stemmer database.
6. Run main.py in the gui folder to start the GUI.
7. Start by clicking on the “Index papers” button. This will automatically call the preprocessing.
8. Once indexing is done, the components can be accessed and interacted with through their corresponding tabs.

## Instructions for the web crawler component

Python 3 and the Scrapy package are required to run the crawler.

Python 3 can be downloaded from https://www.python.org/downloads/.

Install scrapy in a global or virtual environment using the following command via the command line: pip install scrapy.
This will also download and install all of scrapy's dependencies.

Get the crawler source code from the Jur branch of this repository.

Then run the crawler via the command line as follows:
1. cd to the top-level DblpCrawler directory.
2. Use the "scrapy crawl dblpcrawler" command to start the crawler.

The crawler will now run until it is done.
When the crawler finished running it will output some statistics and notify the user.

After running the crawler the crawler_sqlite.db with the scraped data can be found in the data directory.

## Instructions for Topic Modeling and Co Author Graph components.
Topic Modeling is placed in a new gui folder named 'gui_topic_modeling'.

1. For example purpose a data from pregenrated model is already placed in the working directory (src/gui_topic_modeling folder)
2. Run GUI. Enter 9 in the number of topics text field.
3. You can also choose other number but then you have to do create model first before for further analysis.
4. The model already loads files dic.p,corpus.p and author2docModel.p. These can be generated from raw data by running utilities file in topic modeling.
Howvever, these are common for any topic model. So they are generated them before hand and placed them in working directory.
5. GUI is not made for co author graph. But you can explore the functionalities from coauthorgraph.py in TopicModeling folder.


Important Note: If you are cloning from github repository, due to file limitations the sqlite database is not included. 
Please download it from https://www.kaggle.com/benhamner/nips-papers/data and place the database in gui_topic_modeling folder (or from where the main.py is running)
to access the features of topic modeling.
 
 A sample demo can be found at  https://www.youtube.com/watch?v=mdK02BrTQV4&feature=youtu.be

