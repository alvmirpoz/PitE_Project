from django.shortcuts import render

from InsuranceApp import trainingModels
from InsuranceApp import results

# This allow us to train models automatically when server is started
trainingModels.train()
# This allow us to generate the graph comparing the models automatically when server is started
results.generateEntropiesGraph()

def home(request):
    return render(request, 'home.html')

def newPrediction(request):
    return render(request,'newPrediction.html')

def myPrediction(request):
    
    user_attributes = []
    attributes = ['Age', 'Gender', 'Race', 'DrivingExperience', 'Education',
    'Income', 'VehicleOwnership', 'VehicleYear', 'Married', 'Children', 'VehicleType']

    for a in attributes:
        user_attributes.append(request.POST.get(a))

    predictions = results.predictUser(user_attributes)

    return render(request, 'myPrediction.html', {'predictions': predictions})

def learnMore(request):
    return render(request,'learnMore.html')