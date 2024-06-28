from django.shortcuts import render, redirect
from first.models import Post
from django.http import HttpResponse
from first.forms import UpImgForm
from processing.Image_Detection import process_img
from django.contrib.auth import login, logout
from .forms import RegisterUserForm, LoginForm
from django.contrib.auth import authenticate


import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Create your views here.
def first_view(req):
    context = {
        'Plastic':3000,
        'Organic':500,
        'Alumunium':5000
    }

    return render(req, '1st/first.html', context)

def db_acc(req):
    post_data = Post.objects.all()
    return render(req, '1st/first_1.html', {'posts':post_data})

def this_post(req, slug):
    post = Post.objects.get(slug=slug)
    return render(req, '1st/postpage.html', {'post':post})


def upimage_view(request):

    if request.method == 'POST':
        form = UpImgForm(request.POST, request.FILES)
        if form.is_valid():
            form.instance.name = request.user.username
            print(type(request.user.username))
            form.save()
            obj = form.instance
            imgpath = f'../../{obj.Trash_Img.url}'
            path_predict = process_img.add_box(imgpath, request.user)
            form.instance.name = request.user.username
            form.instance.Detect_Img = path_predict[0]
            form.instance.hasil = path_predict[1]
            form.save()
            print(form.instance.tanggal_detect)
            print(str(form.instance.tanggal_detect))
            print(type(form.instance.tanggal_detect))
            return render(request, '1st/upl_img.html', {'form':form, 'obj':obj, 'pred':path_predict[0]})
    else:
        form = UpImgForm()
    return render(request, '1st/upl_img.html', {'form': form})


def success(request):

    return HttpResponse('successfully uploaded')



def register(request):
    if request.method == 'POST':
        form = RegisterUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log in the newly created user
            return redirect('../')  # Redirect to your desired page
    else:
        form = RegisterUserForm()
    return render(request, '1st/register.html', {'form': form})


def masuks(request):
    if request.method == 'POST':
        form = LoginForm()
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('../')
    else:
        form = LoginForm()
    return render(request, '1st/login.html', {'form': form})


def keluar(request):
    logout(request)
    return redirect('first:login')

pathlib.PosixPath = temp