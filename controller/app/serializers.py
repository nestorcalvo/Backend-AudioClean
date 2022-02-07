from django.db import models
from rest_framework import serializers
from .models import Room, SaveFile

class RoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fields =  '__all__'
class CreateRoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fileds = ('guest_can_pause', 'votes_to_skip')

class SaveFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = SaveFile
        fields = '__all__'