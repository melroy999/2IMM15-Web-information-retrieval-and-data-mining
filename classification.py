# The location of the database.sqlite file, used when accessing the database.
_sqlite_file = "../../data/database.sqlite"

#Labels from https://nips.cc/Conferences/2017/CallForPapers
label_names = ["Algorithms", "Probabilistic Methods", "Optimization", "Applications", "Reinforcement Learning and Planning", "Theory", "Neuroscience and Cognitive Science", "Deep Learning", "Data, Competitions, Implementations, and Software"]
label_attributes = ["Active Learning, Bandit Algorithms, Boosting and Ensemble Methods, Classification, Clustering, Collaborative Filtering, Components Analysis (e.g., CCA, ICA, LDA, PCA), Density Estimation, Dynamical Systems, Hyperparameter Selection, Kernel Methods, Large Margin Methods, Metric Learning, Missing Data, Model Selection and Structure Learning, Multitask and Transfer Learning, Nonlinear Dimensionality Reduction and Manifold Learning, Online Learning, Ranking and Preference Learning, Regression, Reinforcement Learning, Relational Learning, Representation Learning, Semi-Supervised Learning, Similarity and Distance Learning, Sparse Coding and Dimensionality Expansion, Sparsity and Compressed Sensing, Spectral Methods, Sustainability, Stochastic Methods, Structured Prediction, and Unsupervised Learning", " Bayesian Nonparametrics, Bayesian Theory, Belief Propagation, Causal Inference, Distributed Inference, Gaussian Processes, Graphical Models, Hierarchical Models, Latent Variable Models, MCMC, Topic Models, and Variational Inference", "Combinatorial Optimization, Convex Optimization, Non-Convex Optimization, and Submodular Optimization", "Audio and Speech Processing, Computational Biology and Bioinformatics, Computational Social Science, Computer Vision, Denoising, Dialog- and/or Communication-Based Learning, Fairness Accountability and Transparency, Game Playing, Hardware and Systems, Image Segmentation, Information Retrieval, Matrix and Tensor Factorization, Motor Control, Music Modeling and Analysis, Natural Language Processing, Natural Scene Statistics, Network Analysis, Object Detection, Object Recognition, Privacy Anonymity and Security, Quantitative Finance and Econometrics, Recommender Systems, Robotics, Signal Processing, Source Separation, Speech Recognition, Systems Biology, Text Analysis, Time Series Analysis, Video, Motion and Tracking, Visual Features, Visual Perception, Visual Question Answering, Visual Scene Analysis and Interpretation, and Web Applications and Internet Data", "Decision and Control, Exploration, Hierarchical RL, Markov Decision Processes, Model-Based RL, Multi-Agent RL, Navigation, and Planning", "Competitive Analysis, Computational Complexity, Control Theory, Frequentist Statistics, Game Theory and Computational Economics, Hardness of Learning and Approximations, Information Theory, Large Deviations and Asymptotic Analysis, Learning Theory, Regularization, Spaces of Functions and Kernels, and Statistical Physics of Learning", "Auditory Perception and Modeling, Brain Imaging, Brain Mapping, Brain Segmentation, Brain--Computer Interfaces and Neural Prostheses, Cognitive Science, Connectomics, Human or Animal Learning, Language for Cognitive Science, Memory, Neural Coding, Neuropsychology, Neuroscience, Perception, Plasticity and Adaptation, Problem Solving, Reasoning, Spike Train Generation, and Synaptic Modulation", "Adversarial Networks, Attention Models, Biologically Plausible Deep Networks, Deep Autoencoders, Efficient Inference Methods, Efficient Training Methods, Embedding Approaches, Generative Models, Interaction-Based Deep Networks, Learning to Learn, Memory-Augmented Neural Networks, Neural Abstract Machines, One-Shot/Low-Shot Learning Approaches, Optimization for Deep Networks, Predictive Models, Program Induction, Recurrent Networks, Supervised Deep Networks, Virtual Environments, and Visualization/Expository Techniques for Deep Networks", " Benchmarks, Competitions or Challenges, Data Sets or Data Repositories, and Software Toolkits"]
#features_data = van Sai his topics

print(label_names)
print(label_attributes[0].split(','))



'''
#DEMO on https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at our data
print(label_names)
print('Class label = ', labels[0])
print(feature_names)
print(features[0])

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
preds = gnb.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))
'''
