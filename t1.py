import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

from gensim.models import AuthorTopicModel


model_list = []
for i in range(5):
    model = AuthorTopicModel(corpus=corpus, num_topics=7, id2word=dictionary.id2token, \
                    author2doc=author2docModel, chunksize=2000, passes=100, gamma_threshold=1e-10, \
                    eval_every=0, iterations=1, random_state=i)
    top_topics = model.top_topics(corpus)
    tc = sum([t[1] for t in top_topics])
    model_list.append((model, tc))
    