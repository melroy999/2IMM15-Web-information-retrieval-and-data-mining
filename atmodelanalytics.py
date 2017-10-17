import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import AuthorTopicModel


model=AuthorTopicModel.load('model.atmodel')
#m.show_topic(5)

n_topics=7
topic_labels=['Topic #'+str(i) for i in range(n_topics)]
for topic in model.show_topics(num_topics=n_topics):
    print('Label: ' + topic_labels[topic[0]])
    words = ''
    for word, prob in model.show_topic(topic[0]):
        words += word + ' '
    print('Words: ' + words)
    print()
    
    
    
    
    
    
    
    
#from gensim.models import atmodel
#doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)
#    
#corpus_words = sum(cnt for document in model.corpus for _, cnt in document)
#
#perwordbound = model.bound(model.corpus, author2doc=model.author2doc, \
#                           doc2author=model.doc2author) / corpus_words
#print(perwordbound)


top_topics = model.top_topics(model.corpus)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
smallest_author = 0  # Ignore authors with documents less than this.
authors = [model.author2id[a] for a in model.author2id.keys() if len(model.author2doc[a]) >= smallest_author]
_ = tsne.fit_transform(model.state.gamma[authors, :])  # Result stored in tsne.embedding_

# Tell Bokeh to display plots inside the notebook.
from bokeh.io import output_notebook
output_notebook()

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource

x = tsne.embedding_[:, 0]
y = tsne.embedding_[:, 1]
author_names = [model.id2author[a] for a in authors]

# Radius of each point corresponds to the number of documents attributed to that author.
scale = 0.1
author_sizes = [len(model.author2doc[a]) for a in author_names]
radii = [size * scale for size in author_sizes]

source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            author_names=author_names,
            author_sizes=author_sizes,
            radii=radii,
        )
    )

# Add author names and sizes to mouse-over info.
hover = HoverTool(
        tooltips=[
        ("author", "@author_names"),
        ("size", "@author_sizes"),
        ]
    )

p = figure(tools=[hover, 'crosshair,pan,wheel_zoom,box_zoom,reset,save,lasso_select'])
p.scatter('x', 'y', radius='radii', source=source, fill_alpha=0.6, line_color=None)
show(p)

