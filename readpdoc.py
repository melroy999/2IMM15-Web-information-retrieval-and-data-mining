import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


#docs=pickle.load(open("pdoc.p","rb")).values.tolist()

# Compute bigrams.
from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)