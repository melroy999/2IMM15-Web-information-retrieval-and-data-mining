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
import A_TM
import findPercMatch


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
    return AuStrength,l


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


def create_doc_citation_matrix(df_papers):
    m=[]
    for i in range(len(df_papers)):
        p=[]
        print(i)
        for j in range(len(df_papers)):
            print(j)
            p.append(findPercMatch.returnPercentageMatch(df_papers['title'][i],df_papers['paper_text'][j]))
        
        p[i]=0
        m.append(p)
    pickle.dump(open("citationMatrix.p","wb"))
    return m
    

def plot_degree_centralty_hist(G):
    plt.hist(list(nx.degree_centrality(G).values()))
    plt.xlabel('Degree Centraility')
    plt.ylabel('Count')
    plt.title('Histogram of Authors degree centraility in NIPS dataset')
    plt.show()

def plot_betweenness_centralty_hist(G):
    plt.hist(list(nx.betweenness_centrality(G).values()))
    plt.xlabel('Betweenness Centraility')
    plt.ylabel('Count')
    plt.title('Histogram of Authors Betweenness centraility in NIPS dataset')
    plt.show()
    
def matrix_plot_largest_connected_component(G):
    largest_ccs = sorted(nx.connected_component_subgraphs(G), key=lambda x: len(x))[-1]
    print(len(largest_ccs.nodes()))
    h = nv.MatrixPlot(largest_ccs)
    h.draw()
    plt.show()

def arc_plot(G):
        # Iterate over all the nodes in G, including the metadata
    for n, d in G.nodes(data=True):
    
        # Calculate the degree of each node: G.node[n]['degree']
        G.node[n]['degree'] = nx.degree(G,n)
        
    # Create the ArcPlot object: a
    a = nv.ArcPlot(G,node_order='degree')
    
    # Draw the ArcPlot to the screen
    a.draw()
    plt.show()

def circos_plot(G):
        # Iterate over all the nodes, including the metadata
    for n, d in G.nodes(data=True):
    
        # Calculate the degree of each node: G.node[n]['degree']
        G.node[n]['degree'] = nx.degree(G,n)
    
    # Create the CircosPlot object: c
    c =nv.CircosPlot(G,node_order='degree')
    
    # Draw the CircosPlot object to the screen
    c.draw()
    plt.show()

def circos_plot_largest_clique(G):
        # Find the author(s) that are part of the largest maximal clique: largest_clique
    largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1]
    
    # Create the subgraph of the largest_clique: G_lc
    G_lc = G.subgraph(largest_clique)
    
    # Create the CircosPlot object: c
    c = nv.CircosPlot(G_lc)
    
    # Draw the CircosPlot to the screen
    c.draw()
    plt.show()

def recommendation(G):
        # Import necessary modules
    from itertools import combinations
    from collections import defaultdict
    
    # Initialize the defaultdict: recommended
    recommended = defaultdict(int)
    
    # Iterate over all the nodes in G
    for n, d in G.nodes(data=True):
    
        # Iterate over all possible triangle relationship combinations
        for n1, n2 in combinations(G.neighbors(n), 2):
        
            # Check whether n1 and n2 do not have an edge
            if not G.has_edge(n1, n2):
            
                # Increment recommended
                recommended[(n1, n2)] += 1
    
    # Identify the top 10 pairs of users
    all_counts = sorted(recommended.values())
    top10_pairs = [pair for pair, count in recommended.items() if count >=5]
    print(top10_pairs)
    return all_counts, recommended,top10_pairs


#model=atm.load_model()
#n_topics=9
#Tm=A_TM.A_TM('ATM'+str(n_topics))
#Tm.load_existing_model()
#    
#G=create_Graph(Tm.df_papers,Tm.df_authors,Tm.model)
As,l=computeAuthorStrength(G)
#gd=get_top_authors(G,top_n=20)
#plot_degree_centralty_hist(G)
#plot_betweenness_centralty_hist(G)
#matrix_plot_largest_connected_component(G)

get_top_collabarators(As,Tm.df_authors['name'].values)
#m=create_doc_citation_matrix(df_papers)
#arc_plot(G)
#circos_plot(G)
#circos_plot_largest_clique(G)
cliques = nx.find_cliques(G)
#print(len(list(cliques)))
c,r,t=recommendation(G)
#c=list(cliques)
#k=[]
#for i in range(len(c)):
#    k.append(len(c[i]))
#
#plt.hist(k)
#plt.xlabel('Clique Size')
#plt.ylabel('Count')
#plt.title('Histogram of Clique sizes')
#plt.show()
cnt=[]
a1=[]
a2=[]
for p in t:
    cnt.append(r[p])
    a1.append(G.node[p[0]]['name'])
    a2.append(G.node[p[1]]['name'])

df=pd.DataFrame({'Author1':a1, 'Author2':a2,'Score':cnt})

    
    
    
    
    
    
    

