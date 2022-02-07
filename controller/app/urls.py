from django.urls import path,re_path
from .views import RoomView, FileUploadView, store_json_data,playAudioFile,retrivePDF,donwloadZIP

urlpatterns = [
    path('room', RoomView.as_view()),
    path('upload', FileUploadView.as_view(), name="upload"),
    # path('upload', upload_file, name="upload"),
    path('parameters',playAudioFile),
    path('process', store_json_data),
    path('processPDF',retrivePDF),
    path('processAudio',playAudioFile),
    path('donwloadZIP', donwloadZIP)
    # path('process', retrive_audios)
]