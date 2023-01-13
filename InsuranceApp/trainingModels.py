import pandas
import numpy
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn import naive_bayes
from sklearn import tree
from matplotlib import pyplot
from sklearn import neighbors



def train():

    dataset = pandas.read_csv('static/dataset.csv', header=0)
    attributes = dataset.loc[:, 'AGE':'VEHICLE_TYPE']
    target = dataset['OUTCOME']

    attributes_coder = preprocessing.OrdinalEncoder()
    attributes_coder.fit(attributes)
    coded_attributes = attributes_coder.transform(attributes)

    target_coder = preprocessing.LabelEncoder()
    coded_target = target_coder.fit_transform(target)


    (training_attributes, test_attributes, training_target, test_target) = model_selection.train_test_split(
        # Datasets to be split, using the same indexes for both
        coded_attributes, coded_target,
        # Random seed value, so that sampling is reproducible,
        # in spite of being random
        random_state=12345,
        # Test set size
        test_size=.33,
        # We stratify with respect to the distribution of values in the target variable
        stratify=target)

    def Naive_Bayes(smoothing):
    
        clasif_NB = naive_bayes.CategoricalNB(alpha=smoothing)  # alpha is the smoothing hyperparameter
        
        # We train the model
        clasif_NB.fit(training_attributes, training_target)

        # We apply the model to the test subset
        predictions = clasif_NB.predict_proba(test_attributes)
        
        cross_entropy = log_loss(test_target, predictions)
        
        return cross_entropy
    
    def Decision_Tree(depth):
    
        clasif_DT = tree.DecisionTreeClassifier(
        max_depth=depth,  # hyperparameter of maximum tree depth
        random_state=12345  # random seed, so that the code can be reproducible
        )

        # We train the model
        clasif_DT.fit(training_attributes, training_target)
        
        # We show the tree resulting from training the model
        pyplot.figure(figsize=(100, 20))  # Width and height of graphics
        tree.plot_tree(clasif_DT,
                        feature_names=attributes_coder.feature_names_in_,
                        class_names=target_coder.classes_)

        # We apply the model to the test subset
        predictions = clasif_DT.predict_proba(test_attributes)
        
        cross_entropy = log_loss(test_target, predictions)
        
        return cross_entropy
    
    def kNN(k, distance_metric):
    
        clasif_kNN = neighbors.KNeighborsClassifier(
        n_neighbors=k,  # hyperparameter of number of neighbors to consider
        metric=distance_metric  # metric hyperparameter for distance calculation
        )

        # We train the model
        clasif_kNN.fit(training_attributes, training_target)

        # We apply the model to the test subset
        predictions = clasif_kNN.predict_proba(test_attributes)
        
        cross_entropy = log_loss(test_target, predictions)
        
        return cross_entropy
    
    models = {}
    models[1] = Naive_Bayes(smoothing=80)
    models[2] = Naive_Bayes(smoothing=90)
    models[3] = Decision_Tree(depth=2)
    models[4] = Decision_Tree(depth=3)
    models[5] = kNN(k=17, distance_metric='hamming')
    models[6] = kNN(k=21, distance_metric='hamming')

    entropies = []

    for e in models.values():
        entropies.append(e)

    models_names = ['Naive Bayes - \nsmoothing 80',
                    'Naive Bayes - \nsmoothing 90',
                    'Decision Tree - \ndepth 2',
                    'Decision Tree - \ndepth 3',
                    'kNN - \n17 neighbors - \nHamming',
                    'kNN - \n21 neighbors - \nHamming']

    fig, axs = pyplot.subplots(1, 1, figsize=(19, 6))
    axs.bar(models_names, entropies)
    fig.suptitle('Binary Cross Entropy')
    for i in range(6):
        if entropies[i] != min(entropies):
            pyplot.text(i-0.15, entropies[i]+0.005, str(entropies[i])[:10])
        else:
            pyplot.text(i-0.20, entropies[i]+0.005, str(entropies[i])[:10], color='green', weight='bold')
    pyplot.show()

    print("Hello World")