from sklearn.metrics import log_loss
from matplotlib import pyplot
import pickle

def predictUser(user_attributes):
    
    # We get the stored models and the attributes coder
    models = pickle.load(open('InsuranceApp/pickled/models.pkl', 'rb'))
    attributes_coder = pickle.load(open('InsuranceApp/pickled/attributes_coder.pkl', 'rb'))

    # We code the user attributes
    user_attributes = attributes_coder.transform([user_attributes])

    predictions = {}

    models_names = ['Naive Bayes - Smoothing 80',
                    'Naive Bayes - Smoothing 90',
                    'Decision Tree - Depth 2',
                    'Decision Tree - Depth 3',
                    'kNN - 17 Neighbors - Hamming',
                    'kNN - 21 Neighbors - Hamming']

    # For each model (and its name):
    for name, model in zip(models_names, models.values()):

        predictions[name] = {}

        # We store the probability of each outcome
        probabilities = model.predict_proba(user_attributes)
        predictions[name]['No'] = round(probabilities[0][0]*100, 2)
        predictions[name]['Yes'] = round(probabilities[0][1]*100, 2)

    return predictions

def generateEntropiesGraph():

    models = pickle.load(open('InsuranceApp/pickled/models.pkl', 'rb'))
    test_attributes = pickle.load(open('InsuranceApp/pickled/test_attributes.pkl', 'rb'))
    test_target = pickle.load(open('InsuranceApp/pickled/test_target.pkl', 'rb'))

    entropies = []
    for m in models.values():
        # We apply the model to the test subset and calculate the binary cross entropy
        predictions = m.predict_proba(test_attributes)
        cross_entropy = log_loss(test_target, predictions)
        entropies.append(cross_entropy)

    models_names = ['Naive Bayes - \nsmoothing 80',
                    'Naive Bayes - \nsmoothing 90',
                    'Decision Tree - \ndepth 2',
                    'Decision Tree - \ndepth 3',
                    'kNN - \n17 neighbors - \nHamming',
                    'kNN - \n21 neighbors - \nHamming']

    # We generate the graph
    fig, axs = pyplot.subplots(1, 1, figsize=(19, 6))
    axs.bar(models_names, entropies)
    fig.suptitle('Binary Cross Entropy')
    fig.set_facecolor('#FFFFFF80')
    for i in range(6):
        if entropies[i] != min(entropies):
            pyplot.text(i-0.15, entropies[i]+0.005, str(entropies[i])[:10])
        else:
            pyplot.text(i-0.20, entropies[i]+0.005, str(entropies[i])[:10], color='green', weight='bold')
    
    # And finally we store it in a png file
    pyplot.savefig('static/img/crossEntropyGraph.png')
