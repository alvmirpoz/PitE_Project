import pandas
from sklearn import preprocessing, model_selection, naive_bayes, tree, neighbors
import pickle

# NAIVE BAYES MODELS
def Naive_Bayes(training_attributes, training_target, smoothing):
    
    # alpha is the smoothing hyperparameter
    clasif_NB = naive_bayes.CategoricalNB(alpha=smoothing)  
    
    # We train the model
    clasif_NB.fit(training_attributes, training_target)
    
    return clasif_NB

# DECISION TREE MODELS
def Decision_Tree(training_attributes, training_target, depth):
    
    clasif_DT = tree.DecisionTreeClassifier(
    max_depth=depth,  # hyperparameter of maximum tree depth
    random_state=12345  # random seed, so that the code can be reproducible
    )

    # We train the model
    clasif_DT.fit(training_attributes, training_target)

    return clasif_DT

# KNN MODELS
def kNN(training_attributes, training_target, k, distance_metric):

    clasif_kNN = neighbors.KNeighborsClassifier(
    n_neighbors=k,  # hyperparameter of number of neighbors to consider
    metric=distance_metric  # metric hyperparameter for distance calculation
    )

    # We train the model
    clasif_kNN.fit(training_attributes, training_target)

    return clasif_kNN

def prepareDataset():

    # Reading data
    dataset = pandas.read_csv('static/dataset.csv', header=0)
    attributes = dataset.loc[:, 'AGE':'VEHICLE_TYPE']
    target = dataset['OUTCOME']

    # Coding data
    attributes_coder = preprocessing.OrdinalEncoder()
    attributes_coder.fit(attributes)
    coded_attributes = attributes_coder.transform(attributes)

    target_coder = preprocessing.LabelEncoder()
    target_coder.fit(target)
    coded_target = target_coder.transform(target)

    # We will later need this trained coder
    pickle.dump(attributes_coder, open('InsuranceApp/pickled/attributes_coder.pkl', 'wb'))

    # Dividing into training and testing subsets
    (training_attributes, test_attributes, training_target, test_target) = model_selection.train_test_split(
        # Datasets to be split
        coded_attributes, coded_target,
        # Random seed value, so that sampling is reproducible, in spite of being random
        random_state=12345,
        # Test subset size
        test_size=.33,
        # We stratify with respect to the distribution of values in the target variable
        stratify=target)
    
    # We store the testing subset into .pkl files
    pickle.dump(test_attributes, open('InsuranceApp/pickled/test_attributes.pkl', 'wb'))
    pickle.dump(test_target, open('InsuranceApp/pickled/test_target.pkl', 'wb'))

    return training_attributes, training_target


def train():

    print("\nTraining models...\n")

    training_attributes, training_target = prepareDataset()
    
    # Training models
    models = {}
    models[1] = Naive_Bayes(training_attributes, training_target, smoothing=80)
    models[2] = Naive_Bayes(training_attributes, training_target, smoothing=90)
    models[3] = Decision_Tree(training_attributes, training_target, depth=2)
    models[4] = Decision_Tree(training_attributes, training_target, depth=3)
    models[5] = kNN(training_attributes, training_target, k=17, distance_metric='hamming')
    models[6] = kNN(training_attributes, training_target, k=21, distance_metric='hamming')

    # Storing trained models into a .pkl file    
    pickle.dump(models, open('InsuranceApp/pickled/models.pkl', 'wb'))
    