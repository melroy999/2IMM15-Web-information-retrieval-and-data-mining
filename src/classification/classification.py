import information_retrieval.vector_space_analysis as vsa
import warnings
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from information_retrieval.indexer import Indexer
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore", UserWarning)

#Labels from https://nips.cc/Conferences/2017/CallForPapers
label_names = ["Algorithms", "Probabilistic Methods", "Optimization", "Applications", "Reinforcement Learning and Planning", "Theory", "Neuroscience and Cognitive Science", "Deep Learning", "Data, Competitions, Implementations, and Software"]
label_attributes = ["Active Learning, Bandit Algorithms, Boosting and Ensemble Methods, Classification, Clustering, Collaborative Filtering, Components Analysis (e.g., CCA, ICA, LDA, PCA), Density Estimation, Dynamical Systems, Hyperparameter Selection, Kernel Methods, Large Margin Methods, Metric Learning, Missing Data, Model Selection and Structure Learning, Multitask and Transfer Learning, Nonlinear Dimensionality Reduction and Manifold Learning, Online Learning, Ranking and Preference Learning, Regression, Reinforcement Learning, Relational Learning, Representation Learning, Semi-Supervised Learning, Similarity and Distance Learning, Sparse Coding and Dimensionality Expansion, Sparsity and Compressed Sensing, Spectral Methods, Sustainability, Stochastic Methods, Structured Prediction, Unsupervised Learning", " Bayesian Nonparametrics, Bayesian Theory, Belief Propagation, Causal Inference, Distributed Inference, Gaussian Processes, Graphical Models, Hierarchical Models, Latent Variable Models, MCMC, Topic Models, Variational Inference", "Combinatorial Optimization, Convex Optimization, Non-Convex Optimization, Submodular Optimization", "Audio and Speech Processing, Computational Biology and Bioinformatics, Computational Social Science, Computer Vision, Denoising, Dialog- and/or Communication-Based Learning, Fairness Accountability and Transparency, Game Playing, Hardware and Systems, Image Segmentation, Information Retrieval, Matrix and Tensor Factorization, Motor Control, Music Modeling and Analysis, Natural Language Processing, Natural Scene Statistics, Network Analysis, Object Detection, Object Recognition, Privacy Anonymity and Security, Quantitative Finance and Econometrics, Recommender Systems, Robotics, Signal Processing, Source Separation, Speech Recognition, Systems Biology, Text Analysis, Time Series Analysis, Video, Motion and Tracking, Visual Features, Visual Perception, Visual Question Answering, Visual Scene Analysis and Interpretation, Web Applications and Internet Data", "Decision and Control, Exploration, Hierarchical RL, Markov Decision Processes, Model-Based RL, Multi-Agent RL, Navigation, Planning", "Competitive Analysis, Computational Complexity, Control Theory, Frequentist Statistics, Game Theory and Computational Economics, Hardness of Learning and Approximations, Information Theory, Large Deviations and Asymptotic Analysis, Learning Theory, Regularization, Spaces of Functions and Kernels, Statistical Physics of Learning", "Auditory Perception and Modeling, Brain Imaging, Brain Mapping, Brain Segmentation, Brain--Computer Interfaces and Neural Prostheses, Cognitive Science, Connectomics, Human or Animal Learning, Language for Cognitive Science, Memory, Neural Coding, Neuropsychology, Neuroscience, Perception, Plasticity and Adaptation, Problem Solving, Reasoning, Spike Train Generation, Synaptic Modulation", "Adversarial Networks, Attention Models, Biologically Plausible Deep Networks, Deep Autoencoders, Efficient Inference Methods, Efficient Training Methods, Embedding Approaches, Generative Models, Interaction-Based Deep Networks, Learning to Learn, Memory-Augmented Neural Networks, Neural Abstract Machines, One-Shot/Low-Shot Learning Approaches, Optimization for Deep Networks, Predictive Models, Program Induction, Recurrent Networks, Supervised Deep Networks, Virtual Environments, Visualization/Expository Techniques for Deep Networks", " Benchmarks, Competitions or Challenges, Data Sets or Data Repositories, Software Toolkits"]
label_attributes_cleaned = ["active", "learning", "bandit", "algorithms", "boosting", "ensemble", "methods", "classification", "clustering", "collaborative", "filtering", "components", "analysis", "cca", "ica", "lda", "pca", "density", "estimation", "dynamical", "systems", "hyperparameter", "selection", "kernel", "methods", "large", "margin", "methods", "metric", "learning", "missing", "data", "model", "selection", "structure", "learning", "multitask", "transfer", "learning", "nonlinear", "dimensionality", "reduction", "manifold", "learning", "online", "learning", "ranking", "preference", "learning", "regression", "reinforcement", "learning", "regression", "reinforcement", "learning", "relational", "learning", "representation learning", "semi", "supervised", "semi-supervised", "learning", "similarity", "distance", "learning", "sparse", "coding", "dimensionality", "expansion", "sparsity", "compressed", "sensing", "spectral", "methods", "sustainability", "stochastic", "methods", "structured", "prediction", "unsupervised", "learning", "bayesian", "nonparametrics", "bayesian", "theory", "belief", "propagation", "causal", "inference", "distributed", "inference", "gaussian", "processes", "graphical", "models", "hierarchical", "models", "latent", "variable", "models", "mcmc", "topic", "models", "variational", "inference", "combinatorial", "optimization", "convex", "optimization", "non-convex", "optimization", "submodular", "optimization", "audio", "speech", "processing", "computational", "biology", "bioinformatics", "computational", "social", "science", "computer", "vision", "denoising", "dialog", "communication-based", "communication", "based" "learning", "fairness", "accountability", "transparency", "game", "playing", "hardware", "systems", "image", "segmentation", "information", "retrieval", "matrix", "tensor", "factorization", "motor", "control", "music", "modeling", "analysis", "natural", "language", "processing", "natural", "scene", "statistics", "network", "analysis", "object", "detection", "object", "recognition", "privacy", "anonymity", "security", "quantitative", "finance", "econometrics", "recommender", "systems", "robotics", "signal", "processing", "source", "separation", "speech", "recognition", "systems", "biology", "text", "analysis", "time", "series", "analysis", "video", "motion", "tracking", "visual", "features", "visual", "perception", "visual", "question", "answering", "visual", "scene", "analysis", "interpretation", "web", "applications", "internet", "data", "decision", "control", "exploration", "hierarchical", "rl", "markov", "decision", "processes", "model", "based", "rl", "multi", "agent", "rl", "navigation", "planning", "competitive", "analysis", "computational", "complexity", "control", "theory", "frequentist", "statistics", "game", "theory", "computational", "cconomics", "hardness", "learning", "approximations", "information", "theory", "large", "deviations", "asymptotic", "analysis", "learning", "theory", "regularization", "spaces", "functions" "kernels", "statistical", "physics", "learning", "auditory", "perception", "modeling", "brain", "imaging", "brain", "mapping", "brain", "segmentation", "brain", "computer", "interfaces", "neural", "prostheses", "cognitive", "science", "connectomics", "human", "animal", "learning", "language", "cognitive", "science", "memory", "neural", "coding", "neuropsychology", "neuroscience", "perception", "plasticity", "adaptation", "problem", "solving", "reasoning", "spike", "train", "generation", "synaptic", "modulation", "adversarial", "networks", "attention", "models", "biologically", "plausible", "deep", "networks", "deep", "autoencoders", "efficient", "inference", "methods", "efficient", "training", "methods", "embedding", "approaches", "generative", "models", "interaction", "based", "deep", "networks", "learning", "learn", "memory", "augmented", "neural", "networks", "neural", "abstract", "machines", "one", "shot", "low", "shot", "learning", "approaches", "optimization", "deep", "networks", "predictive", "models", "program", "induction", "recurrent", "networks", "supervised", "deep", "networks", "virtual", "environments", "visualization", "expository", "techniques", "deep", "networks", "benchmarks", "competitions", "challenges", "data", "sets", "data", "repositories", "software", "toolkits"]

#Labels as numbers:
#Algorithms = 0

#print(label_names)
#i = 0
#for attr_set in label_attributes:
#    print('label = ', label_names[i])
#    i += 1
#    for feature in attr_set.split(','):
#        print(feature)

result_count = 8
indexer = None
already_indexed = False
already_trained = False
onevsrest_classifier = None

array_labels = []
add_labels = []
# This is the data structure that contains the attributes
data = []
# This is the ground truth for each paper
ground_truth = []


def search_vector_query(query, indexer):
    target_field = "paper_text"

    try:
        scores = vsa.search(query, indexer, target_field, scoring_measure_name="tf", similar_document_search=False,
           similarity_measure_name="Cosine coefficient")

        if scores is not None:
            # Print the scores.
            #print(query, scores, result_count)
            return query, scores, result_count
    except vsa.EmptyQueryException:
        print("Query is empty after normalization, please change the query.")


def find_labels():
    start_indexing()
    for attr_set in label_attributes:
        one_label = []
        for feature in attr_set.split(','):
            one_feature = []
            _, id, _ = search_vector_query(feature, indexer)
            for x in range(0, result_count-1):
                if (id[x][1] > 0.1):
                    #print(id[x][0])
                    one_feature.append(id[x][0])
                    #print(one_feature)
            for y in range(0, len(one_feature)):
                one_label.append(one_feature[y])
            #print(one_label, "\n")
        add_labels.append(one_label)
    #real_labels = []
    for z in range(0, 9):
        array_labels.append(add_labels[z])
        #print(add_labels[z], "\n")
     #   for j in range(0, len(add_labels[y])):
      #      for number in add_labels[y][j]:
       #         real_labels.append(number)


def fit_data():
    global indexer
    # For each label
    print("Fitting training data")
    for i in range(0, 9):
        for paper in array_labels[i]:
            results = indexer.results["papers"]["paper_text"][paper]["tf"]

            #Create a attribute array with tf for one paper
            temp = []
            for var in label_attributes_cleaned:
                if results.get(var) == None:
                    temp.append(0)
                else:
                    temp.append(results.get(var))
            data.append(temp)
            ground_truth.append(i)
            #print(temp)


#print(results.get("memory"))
#sorted_results = sorted(results.items(), key=operator.itemgetter(1))
#print(sorted_results)

# X = [[1, 2, 3, 4],
#      [4, 2, 3, 1],
#      [1, 1, 1, 1],
#      [4, 4, 2, 1]]
# y = [1, 2, 3, 4]


def start_indexing():
    global already_indexed
    global indexer
    if not already_indexed:
        print("Indexing corpus, this usually takes around 5 seconds")
        indexer = Indexer(None)
        indexer.index_corpus("None", False)
        already_indexed = True


def train_classifier():
    global already_trained
    global onevsrest_classifier
    if not already_trained:
        print("Preparing papers for training oneVSrest Classifier...(can take up to 1 minute)")
        find_labels()
        fit_data()
        print("Training the oneVSrest Classifier...")
        onevsrest_classifier = OneVsRestClassifier(SVC(kernel='linear'))
        onevsrest_classifier.fit(data, ground_truth)
        already_trained = True


def print_results():
    X_train, X_test, y_train, y_test = train_test_split(data, ground_truth, test_size=.25, random_state=42, shuffle=True)

    classifier = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
         verbose=0)
    print("LinearSVC:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = OneVsRestClassifier(SVC(kernel='linear'))
    print("OneVsRestClassifier:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    print("BernoulliNB:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = GaussianNB(priors=None)
    print("GaussianNB:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
    print("SGDClassifier:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    print("KNeighborsClassifier:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)

    classifier = Perceptron(max_iter=1000)
    print("Perceptron:")
    print_pred_acc(classifier, X_train, X_test, y_train, y_test)


def print_pred_acc(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    print(pred)
    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy: ", score)
    print()


def predict_label(paperID):
    #make sure that indexing step is done
    start_indexing()
    # make sure that training step is done
    train_classifier()

    temp = []
    results = indexer.results["papers"]["paper_text"][paperID]["tf"]
    for var in label_attributes_cleaned:
        if results.get(var) is None:
            temp.append(0)
        else:
            temp.append(results.get(var))

    predicted_label = int(onevsrest_classifier.predict([temp]))
    if predicted_label == 0:
        return "Algorithms"
    elif predicted_label == 1:
        return "Probabilistic Methods"
    elif predicted_label == 2:
        return "Optimization"
    elif predicted_label == 3:
        return "Applications"
    elif predicted_label == 4:
        return "Reinforcement Learning and Planning"
    elif predicted_label == 5:
        return "Theory"
    elif predicted_label == 6:
        return "Neuroscience and Cognitive Science"
    elif predicted_label == 7:
        return "Deep Learning"
    elif predicted_label == 8:
        return "Data, Competitions, Implementations, and Software"
    else:
        raise Exception("The classifier returned an invalid label!")

# Execute this for training the oneVSrest Classifier with all labeled data and predict a label from a given paperID
#print("Prediction label for a paper:", predict_label(500))


# Execute this for training all the classifiers and return their accuracy
# find_labels()
# fit_data()
# print_results()
