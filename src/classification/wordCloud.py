from information_retrieval.indexer import Indexer
from sklearn.svm import LinearSVC

label_names = ["Algorithms", "Probabilistic Methods", "Optimization", "Applications", "Reinforcement Learning and Planning", "Theory", "Neuroscience and Cognitive Science", "Deep Learning", "Data, Competitions, Implementations, and Software"]
label_attributes = ["Active Learning, Bandit Algorithms, Boosting and Ensemble Methods, Classification, Clustering, Collaborative Filtering, Components Analysis (e.g., CCA, ICA, LDA, PCA), Density Estimation, Dynamical Systems, Hyperparameter Selection, Kernel Methods, Large Margin Methods, Metric Learning, Missing Data, Model Selection and Structure Learning, Multitask and Transfer Learning, Nonlinear Dimensionality Reduction and Manifold Learning, Online Learning, Ranking and Preference Learning, Regression, Reinforcement Learning, Relational Learning, Representation Learning, Semi-Supervised Learning, Similarity and Distance Learning, Sparse Coding and Dimensionality Expansion, Sparsity and Compressed Sensing, Spectral Methods, Sustainability, Stochastic Methods, Structured Prediction, Unsupervised Learning", " Bayesian Nonparametrics, Bayesian Theory, Belief Propagation, Causal Inference, Distributed Inference, Gaussian Processes, Graphical Models, Hierarchical Models, Latent Variable Models, MCMC, Topic Models, Variational Inference", "Combinatorial Optimization, Convex Optimization, Non-Convex Optimization, Submodular Optimization", "Audio and Speech Processing, Computational Biology and Bioinformatics, Computational Social Science, Computer Vision, Denoising, Dialog- and/or Communication-Based Learning, Fairness Accountability and Transparency, Game Playing, Hardware and Systems, Image Segmentation, Information Retrieval, Matrix and Tensor Factorization, Motor Control, Music Modeling and Analysis, Natural Language Processing, Natural Scene Statistics, Network Analysis, Object Detection, Object Recognition, Privacy Anonymity and Security, Quantitative Finance and Econometrics, Recommender Systems, Robotics, Signal Processing, Source Separation, Speech Recognition, Systems Biology, Text Analysis, Time Series Analysis, Video, Motion and Tracking, Visual Features, Visual Perception, Visual Question Answering, Visual Scene Analysis and Interpretation, Web Applications and Internet Data", "Decision and Control, Exploration, Hierarchical RL, Markov Decision Processes, Model-Based RL, Multi-Agent RL, Navigation, Planning", "Competitive Analysis, Computational Complexity, Control Theory, Frequentist Statistics, Game Theory and Computational Economics, Hardness of Learning and Approximations, Information Theory, Large Deviations and Asymptotic Analysis, Learning Theory, Regularization, Spaces of Functions and Kernels, Statistical Physics of Learning", "Auditory Perception and Modeling, Brain Imaging, Brain Mapping, Brain Segmentation, Brain--Computer Interfaces and Neural Prostheses, Cognitive Science, Connectomics, Human or Animal Learning, Language for Cognitive Science, Memory, Neural Coding, Neuropsychology, Neuroscience, Perception, Plasticity and Adaptation, Problem Solving, Reasoning, Spike Train Generation, Synaptic Modulation", "Adversarial Networks, Attention Models, Biologically Plausible Deep Networks, Deep Autoencoders, Efficient Inference Methods, Efficient Training Methods, Embedding Approaches, Generative Models, Interaction-Based Deep Networks, Learning to Learn, Memory-Augmented Neural Networks, Neural Abstract Machines, One-Shot/Low-Shot Learning Approaches, Optimization for Deep Networks, Predictive Models, Program Induction, Recurrent Networks, Supervised Deep Networks, Virtual Environments, Visualization/Expository Techniques for Deep Networks", " Benchmarks, Competitions or Challenges, Data Sets or Data Repositories, Software Toolkits"]
label_attributes_cleaned = ["active", "learning", "bandit", "algorithms", "boosting", "ensemble", "methods", "classification", "clustering", "collaborative", "filtering", "components", "analysis", "cca", "ica", "lda", "pca", "density", "estimation", "dynamical", "systems", "hyperparameter", "selection", "kernel", "methods", "large", "margin", "methods", "metric", "learning", "missing", "data", "model", "selection", "structure", "learning", "multitask", "transfer", "learning", "nonlinear", "dimensionality", "reduction", "manifold", "learning", "online", "learning", "ranking", "preference", "learning", "regression", "reinforcement", "learning", "regression", "reinforcement", "learning", "relational", "learning", "representation learning", "semi", "supervised", "semi-supervised", "learning", "similarity", "distance", "learning", "sparse", "coding", "dimensionality", "expansion", "sparsity", "compressed", "sensing", "spectral", "methods", "sustainability", "stochastic", "methods", "structured", "prediction", "unsupervised", "learning", "bayesian", "nonparametrics", "bayesian", "theory", "belief", "propagation", "causal", "inference", "distributed", "inference", "gaussian", "processes", "graphical", "models", "hierarchical", "models", "latent", "variable", "models", "mcmc", "topic", "models", "variational", "inference", "combinatorial", "optimization", "convex", "optimization", "non-convex", "optimization", "submodular", "optimization"]

indexer = Indexer(None)
indexer.index_corpus("None", True)

results = indexer.results["papers"]["paper_text"][1]["tf"]
print("The results: ", results)

data = []
for var in label_attributes_cleaned:
    if results.get(var) is None:
        data.append(0)
    else:
        data.append(results.get(var))

print(data)

#print(results.get("memory"))
#sorted_results = sorted(results.items(), key=operator.itemgetter(1))
#print(sorted_results)

X = [[1, 2, 3, 4],
     [4, 2, 3, 1],
     [1, 1, 1, 1],
     [4, 4, 2, 1]]
y = [1, 2, 3, 4]
clf = LinearSVC(random_state=0)
clf.fit(X, y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=0)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[4, 4, 2, 1]]))