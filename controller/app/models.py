from django.db import models
from django.db.models import Model, CharField,BooleanField,IntegerField,DateTimeField,FileField,JSONField
import string
from django import forms
import random
from django.db.models.deletion import CASCADE, SET_NULL
from django.db.models.fields import BLANK_CHOICE_DASH
import os

from django.urls.conf import path

def generate_code():
    length = 6
    while True:
        code = ''.join(random.choices(string.ascii_uppercase,k=length))
        if Room.objects.filter(code = code).count() == 0:
            break
    return code
def user_directory_path(instance, filename):
    return f"posts/{instance.path.token}/{filename}"
# Create your models here.
class Room(Model):
    code = CharField(max_length=8, default=generate_code, unique= True)
    host = CharField(max_length=50, unique=True)
    guest_can_pause = BooleanField(null=False, default=False)
    votes_to_skip = IntegerField(null=False, default=1)
    created_at = DateTimeField(auto_now_add=True)
    
class FolderUser(Model):
    token = CharField(max_length=50, unique=True, default="")
class SaveFile(Model):
    file = FileField(upload_to=user_directory_path, null=False)
    path = models.ForeignKey(FolderUser, on_delete=SET_NULL,null=True, verbose_name='Path')
    
    

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()