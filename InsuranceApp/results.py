from sklearn.metrics import log_loss
from matplotlib import pyplot
import pickle

def predictUser(user_attributes):
    
    models = pickle.load(open('InsuranceApp/pickled/models.pkl', 'rb'))

    predictions = []
    probabilities = []

    for m in models.values():
        predictions.append(m.predict(user_attributes))
        probabilities.append(m.predict_proba(user_attributes))

    return predictions, probabilities

def generateEntropiesGraph():

    models = pickle.load(open('InsuranceApp/pickled/models.pkl', 'rb'))
    test_attributes = pickle.load(open('InsuranceApp/pickled/test_attributes.pkl', 'rb'))
    test_target = pickle.load(open('InsuranceApp/pickled/test_target.pkl', 'rb'))

    entropies = []
    for m in models.values():
        # We apply the model to the test subset
        predictions = m.predict_proba(test_attributes)
        cross_entropy = log_loss(test_target, predictions)
        entropies.append(cross_entropy)

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
