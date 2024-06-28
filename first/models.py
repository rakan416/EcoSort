from typing import Any
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, UserManager, PermissionsMixin
from django.utils import timezone

# Create your models here.
class Post(models.Model):
    title = models.CharField(max_length=75)
    body = models.TextField()
    slug = models.SlugField()
    date = models.DateTimeField(auto_now_add=True)
    banner = models.ImageField(default='default.jpg', blank=True, upload_to='images/')

    def __str__(self) -> str:
        return self.title

class UpImg(models.Model):
    name = models.CharField(max_length=50, blank=False, null=False, default='Anonim')
    Trash_Img = models.ImageField(upload_to='images/')
    Detect_Img = models.ImageField(default='default.png')
    tanggal_detect = models.DateTimeField(default=timezone.now())
    hasil = models.TextField(default='(no detections)')


class CustomUserManager(UserManager):
    def _create_user(self, username, password, **extra_fields):
        
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save()
        return user
    
    def create_user(self, username: str, password: str | None = ..., **extra_fields: Any) -> Any:
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(username, password, **extra_fields)
    
    def create_superuser(self, username: str, password: str | None, **extra_fields: Any) -> Any:
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self._create_user(username, password, **extra_fields)
    
class User(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(unique=True, max_length=50)
    first_name = models.CharField(max_length=50, default='Orgil')
    last_name = models.CharField(max_length=50, default='')
    province = models.CharField(max_length=50, default='Jawa Timur')
    city = models.CharField(max_length=50, default='Surabaya')

    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    
    date_joined = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(blank=True, null=True)

    objects = CustomUserManager()
    USERNAME_FIELD = 'username'
    EMAIL_FIELD = None
    REQUIRED_FIELDS = []

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def get_full_name(self):
        return self.first_name + self.last_name
    
    def get_short_name(self):
        return self.first_name

