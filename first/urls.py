from django.urls import path

from first import views

app_name = 'first'

urlpatterns = [
    # path('', views.first_view, name='firstpage'),
    path('', views.upimage_view, name='image_upload'),
    path('trydb/', views.db_acc, name='db_acc'),
    # path('image_upload', views.upimage_view, name='image_upload'),
    path('success', views.success, name='success'),
    path('register', views.register, name='register'),
    path('login', views.masuks, name='login'),
    path('logout', views.keluar, name='out'),
]