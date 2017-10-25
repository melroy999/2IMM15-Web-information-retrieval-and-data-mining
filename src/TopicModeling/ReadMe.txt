1. For example purpose a data from pregenrated model is already placed in the working directory (gui folder)
2. Run GUI. Select Topic Modeling tab and enter 9 in the number of topics text field.
3. You can also choose other number but then you have to do create model first before for further analysis.
4. The model already loads files dic.p,corpus.p and author2docModel.p. These can be generated from raw data by running utilities file in topic modeling.
Howvever, these are common for any topic model we generated them before hand and placed them in working directory.
5. GUI is not made for co author graph. But you can explore the functionalities from coauthorgraph.py in TopicModeling folder.


Important Note: If you are cloning from github repository, due to file limitations the sqlite database is not included. 
Please download it from https://www.kaggle.com/benhamner/nips-papers/data and place the database in gui folder (or from where the main.py is running)
to access the features of topic modeling.
