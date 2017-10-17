import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

import networkx as nx
import nxviz as nv

from sqlalchemy import create_engine
import pickle
import itertools
#visualization packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
import atm


#df_a2doc=pickle.load(open("a2doc.p","rb"))
#df_doc2a=pickle.load(open("doc2a.p","rb"))
#
# 
#database = "sqlite:///database.sqlite"
# 
## create a database connection
#engine=create_engine(database)
#conn = engine.connect()
##with conn:
#df_papers=pd.read_sql_table('papers',conn) 
#df_authors=pd.read_sql_table('authors',conn) 
#df_map=pd.read_sql_table('paper_authors',conn) 
#
#df_papers.id=df_papers.id.astype(str)
#df_map.id=df_map.id.astype(str)
#df_map.paper_id=df_map.paper_id.astype(str)
#df_map.author_id=df_map.author_id.astype(str)
#df_authors.id=df_authors.id.astype(str)
#
#
#a_id=df_authors['id'].values
#p_id=df_papers['id'].values
#
#
#


def create_Graph(df_papers,df_authors,atmodel):
    G=nx.Graph()
    nx.set_node_attributes(G,'name','Author_Name')
    nx.set_node_attributes(G,'docs','Author_Docs')
    
    G.add_nodes_from(df_authors['id'].values)
    a_name=df_authors['name'].values
    nodes=G.nodes()
    doc2a=list(atmodel.doc2author.values())
    for  i in range(len(G.nodes())):
        G.node[nodes[i]]['name']=a_name[i]
        G.node[nodes[i]]['docs']=atmodel.author2doc[nodes[i]]
        
    
    for i in range(len(doc2a)):
        ca=doc2a[i]
        k=list(itertools.permutations(ca, 2))
        if(len(k)>0):
            for i in range(len(k)):
                s=k[i][0]
                d=k[i][1]
                if G.has_edge(s,d):
                    G[s][d]['wt']+=0.5
                else:
                    G.add_edge(s,d)
                    G[s][d]['wt']=0.5
    print('Graph Created')
    return G
                
                
#for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]['wt']):
#    print('{a} {b} {w}'.format(a=a, b=b, w=data['wt']))
    
    
def computeAuthorStrength(G):
    AuStrength=[]
    a_id=G.nodes()
    for i in range(len(a_id)):
        ca=G.neighbors(a_id[i])
        
        AuStrength.append(np.sum([G[a_id[i]][d]['wt'] for d in G.neighbors(a_id[i])]))
    
    sg = list(nx.connected_component_subgraphs(G))
    
    l=[]
    for i in range(len(sg)):
        s=sg[i].nodes()
        l.append(len(s))
        return AuStrength


def get_top_collabarators(Strength,a_names,top_n=10):
    AS=np.array(Strength)
    top_idx=np.flip(np.argsort(AS)[-top_n:],axis=0)
    n=[AS[top_idx[i]] for i in range(len(top_idx))]
    an=[a_names[top_idx[i]] for i in range(len(top_idx))]
    df=pd.DataFrame({'Name':an,'count':n})
    df.plot(x='Name',y='count',kind='bar') 
    plt.show()
    return df

def get_top_authors(G,top_n=10):
    nodes=G.nodes()
    n=[len(G.node[nodes[i]]['docs']) for i in range(len(G.nodes()))]
    top_idx=np.flip(np.argsort(n)[-top_n:],axis=0)
    cn=[n[top_idx[i]] for i in range(len(top_idx))]
    
    an=[G.node[nodes[top_idx[i]]]['name'] for i in range(len(top_idx))]
    df=pd.DataFrame({'Name':an,'count':cn})
    df.plot(x='Name',y='count',kind='bar') 
    plt.show()
    return df

    
    
    

def plot_matrix_Plot(G):    
    nv.MatrixPlot(G).draw()
    plt.show()

#model=atm.load_model()
#G=create_Graph(df_papers,df_authors,model)
#As=computeAuthorStrength(G)
gd=get_top_authors(G,top_n=20)
#get_top_collabarators(As,df_authors['name'].values)

    
    
    

