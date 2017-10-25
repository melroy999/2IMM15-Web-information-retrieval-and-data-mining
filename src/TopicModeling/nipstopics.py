import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import LDA_TM


algorithms='Active Learning, Bandit Algorithms, Boosting and Ensemble Methods, Classification, Clustering, Collaborative Filtering, Components Analysis CCA, ICA, LDA, PCA, Density Estimation, Dynamical Systems, Hyperparameter Selection, Kernel Methods, Large Margin Methods, Metric Learning, Missing Data, Model Selection and Structure Learning, Multitask and Transfer Learning, Nonlinear Dimensionality Reduction and Manifold Learning, Online Learning, Ranking and Preference Learning, Regression, Reinforcement Learning, Relational Learning, Representation Learning, Semi-Supervised Learning, Similarity and Distance Learning, Sparse Coding and Dimensionality Expansion, Sparsity and Compressed Sensing, Spectral Methods, Sustainability, Stochastic Methods, Structured Prediction, and Unsupervised Learning'

probabalistic_methods='Bayesian Nonparametrics, Bayesian Theory, Belief Propagation, Causal Inference, Distributed Inference, Gaussian Processes, Graphical Models, Hierarchical Models, Latent Variable Models, MCMC, Topic Models, and Variational Inference.'

optimization='Combinatorial Optimization, Convex Optimization, Non-Convex Optimization, and Submodular Optimization'

Applications='Audio and Speech Processing, Computational Biology and Bioinformatics, Computational Social Science, Computer Vision, Denoising, Dialog- and/or Communication-Based Learning, Fairness Accountability and Transparency, Game Playing, Hardware and Systems, Image Segmentation, Information Retrieval, Matrix and Tensor Factorization, Motor Control, Music Modeling and Analysis, Natural Language Processing, Natural Scene Statistics, Network Analysis, Object Detection, Object Recognition, Privacy Anonymity and Security, Quantitative Finance and Econometrics, Recommender Systems, Robotics, Signal Processing, Source Separation, Speech Recognition, Systems Biology, Text Analysis, Time Series Analysis, Video, Motion and Tracking, Visual Features, Visual Perception, Visual Question Answering, Visual Scene Analysis and Interpretation, and Web Applications and Internet Data'

Reinforcement_learning_and_planning='Decision and Control, Exploration, Hierarchical RL, Markov Decision Processes, Model-Based RL, Multi-Agent RL, Navigation, and Planning'

Theory='Competitive Analysis, Computational Complexity, Control Theory, Frequentist Statistics, Game Theory and Computational Economics, Hardness of Learning and Approximations, Information Theory, Large Deviations and Asymptotic Analysis, Learning Theory, Regularization, Spaces of Functions and Kernels, and Statistical Physics of Learning'

neuroscience='Auditory Perception and Modeling, Brain Imaging, Brain Mapping, Brain Segmentation, Brain--Computer Interfaces and Neural Prostheses, Cognitive Science, Connectomics, Human or Animal Learning, Language for Cognitive Science, Memory, Neural Coding, Neuropsychology, Neuroscience, Perception, Plasticity and Adaptation, Problem Solving, Reasoning, Spike Train Generation, and Synaptic Modulation'

deep_learning= "Adversarial Networks, Attention Models, Biologically Plausible Deep Networks, Deep Autoencoders, Efficient Inference Methods, Efficient Training Methods, Embedding Approaches, Generative Models, Interaction-Based Deep Networks, Learning to Learn, Memory-Augmented Neural Networks, Neural Abstract Machines, One-Shot/Low-Shot Learning Approaches, Optimization for Deep Networks, Predictive Models, Program Induction, Recurrent Networks, Supervised Deep Networks, Virtual Environments, and Visualization Expository Techniques for Deep Networks "

software='Benchmarks, Competitions or Challenges, Data Sets or Data Repositories, and Software Toolkits'

docs=[algorithms,probabalistic_methods,optimization,Applications,Reinforcement_learning_and_planning,Theory,neuroscience,deep_learning,software]
labels=['Algorithms','Probabalistic Methods','optimization','Applications','Reinforcement Learning and Planning','Theory','Neuroscience and Cognitive Science','Deep Learning','Data Competiotions Implemetations and Software']

n_topics=10
Tm=LDA_TM.LDA_TM('LDA'+str(n_topics))
Tm.load_existing_model()
    
