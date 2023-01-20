from django.shortcuts import render

# Create your views here.


res = None

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request,'predict.html')

def stats(request):
    return 0

def newInsured(request):
    return render(request,'newInsured.html')
#     if request.method == 'POST':
#         form = TourismForm(request.POST)
        
#         if form.is_valid():

#             year =     form.cleaned_data['year']
#             duration = form.cleaned_data['duration']
#             spends = form.cleaned_data['spends'] / 1000
#             mode = int(form.cleaned_data['mode'])
#             purpose = int(form.cleaned_data['purpose'])
#             quarter =  int(form.cleaned_data['quarter'])
#             country =  int(form.cleaned_data['country'])

#             x = [quarter, mode, purpose, year, duration, country, spends, 0.38]
#             global res
#             res = logic_layer(x)
#             return redirect("/predict")
#         else:
#             problem = form.errors.as_data()
#             # This section is used to handle invalid data 
#             messages.error(request, list(list(problem.values())[0][0])[0])
#             form = TourismForm()
#     form = TourismForm()
#     return render(request=request, template_name='main/index2.html', context={"form": form})


def about(request):
    return render(request=request, 
            template_name="about.html")


# def under_construction(request):
#     messages.info(request, "This page coming soon..")
#     return render(request=request, 
#             template_name="main/under_construction.html")
