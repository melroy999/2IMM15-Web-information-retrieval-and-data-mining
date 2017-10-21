import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk


import LDA_TM
p=[]
for i in range(3,25):
    Test=LDA_TM.LDA_TM('LDA'+str(i+1))
    Test.create_LDA_model(i+1)
    p.append(Test.per_word_bound)
