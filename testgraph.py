import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

import networkx as nx

from sqlalchemy import create_engine
 
import itertools

    


G=nx.Graph()
G.add_nodes_from(range(6))
G.add_edge(1,2)
G.add_edge(2,3)

G.add_edge(3,1)
G.add_edge(3,4)

G.add_edge(4,0)
G.add_edge(5,0)
G.add_edge(5,4)

    


nx.draw(G)

plt.show()

print(nx.clustering(G))
sg = list(nx.connected_component_subgraphs(G))