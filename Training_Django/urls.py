"""
URL configuration for Training_Django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.conf.urls.static import static
from django.conf import settings

from Training_Django import views

app_name = 'main'

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', views.index),
    # path('/', views.index),
    # path('home/', views.index1),
    path('', views.homepage, name='home'),
    path('dashboard/',views.dashboard),
    path('history/',views.history),
    path('history/history2', views.history2),
    path('history/', views.history, name='history'),
    path('history2/', views.history2, name='history2'),
    path('homepage/', views.homepage),
    path('history2/', views.history2),
    path('first/' , include('first.urls'))
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)