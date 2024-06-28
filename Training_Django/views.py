from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from first.models import UpImg
import os

import first.models


# method view
def index(req):
    return HttpResponse(
        """<h1>Wellcome to my project!</h1>
        <p><a href="/home">Go to HOMEPAGE</a></p>"""
        )

def index1(req):
    return render(req, 'index.html')

def homepage(request):
    # return HttpResponse("Hello World, I'm Home.")
    return render(request, 'home.html')

@login_required(login_url='first:login')
def dashboard(request):
    # return HttpResponse("My Dashboard page.")
    return render(request, 'dashboard.html')

@login_required(login_url='first:login')
def history(request):
    image_folder = 'media'
    datelist = []
    obj = UpImg.objects.all().values()
    user_hist = {}
    for hist in list(obj):
        if  hist['name'] == request.user.username:
            date = hist['tanggal_detect'].date()
            if date not in user_hist.keys():
                datelist.append(date)
                user_hist.update({date:[hist]})
            else:
                user_hist[date].append(hist)

    return render(request, 'history.html', {'datelist': datelist, 'user_hist':user_hist})

@login_required(login_url='first:login')
def history2(request):
    image_name = request.GET.get('image')
    return render(request, 'history2.html', {'image_name': image_name})