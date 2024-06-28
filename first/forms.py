from django import forms
from django.contrib.auth.forms import UserCreationForm
from first.models import UpImg, User


class UpImgForm(forms.ModelForm):

    class Meta:
        model = UpImg
        fields = ['Trash_Img']


class RegisterUserForm(UserCreationForm):
    first_name = forms.CharField(max_length=50, label='Nama Depan')
    last_name = forms.CharField(max_length=50, label='Nama Belakang')
    province = forms.CharField(max_length=30, label='Privinsi')
    city = forms.CharField(max_length=30, label='Kota')

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'username', 'province', 'city', 'password1', 'password2')


class LoginForm(forms.Form):
  username = forms.CharField(label="Username", max_length=255)
  password = forms.CharField(label="Password", widget=forms.PasswordInput)

  class Meta:
      fields = ('username', 'password')
