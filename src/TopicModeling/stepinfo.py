import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk


def step_info(t,yout):
    print("OS: %f%s"%((yout.max()/yout[-1]-1)*100,'%'))
    print("Tr: %fs"%(t[next(i for i in range(0,len(yout)-1) if yout[i]>yout[-1]*.90)]-t[0]))
    print("Ts: %fs"%(t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1])>3)]-t[0]))


p=k[4:]
t=np.array([i for i in range(len(p))])

step_info(t,np.array(p))