from django.shortcuts import render
from django.core.files.storage import FileSystemStorage, Storage
from rest_framework import generics, serializers, status
from .serializers import RoomSerializer, CreateRoomSerializer, SaveFileSerializer
from .models import Room, SaveFile, UploadFileForm,FolderUser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser, MultiPartParser, FormParser
from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser
from rest_framework.exceptions import ParseError
from ORCA_CLEAN.predict import predict
from django.http import HttpResponse
import os
from django.core.files import File
from ast import literal_eval
import json
from django.core.files.storage import FileSystemStorage
from urllib import parse
from pydub import AudioSegment
from math import ceil
import zipfile
import io
from django.views.decorators.csrf import csrf_exempt

@api_view(['POST'])
def donwloadZIP(request):
    print('Request to extract zip',type(literal_eval(request.POST.get('regions'))))
    print(os.path.dirname(os.path.dirname(__file__)))
    elements = [os.path.dirname(os.path.dirname(__file__)),'media','posts',str(request.POST.get('tokens')), 'Clean']
    Base_path = os.path.join(*elements)
    filenames = []
    for item in literal_eval(request.POST.get('regions')):

        filenames.append(os.path.join(Base_path,'denoised_0_'+item+'.wav'))
        filenames.append(os.path.join(Base_path,'net_input_spec_0_'+item+'.pdf'))
        filenames.append(os.path.join(Base_path,'net_out_spec_0_'+item+'.pdf'))

    buffer = io.BytesIO()
    zip_file = zipfile.ZipFile(buffer, 'w')
    
    # zip_file = zipfile.ZipFile(response, 'w')
    for filename in filenames:
        fdir, fname = os.path.split(filename)
        zip_file.write(filename,fname)
        # open(filename, 'rb').read()
    zip_file.close()
    response = HttpResponse(buffer.getvalue())
    response['Content-Type'] = 'application/x-zip-compressed'
    response['Content-Disposition'] = 'attachment; filename=album.zip'
    # response['Content-Disposition'] = 'attachment; filename={}'.format('files.zip')
    # print(response)
    # zip_file.close()
    return response
   
    # return resp

def split_audio(audio, data_array, path,audio_name):
    print(len(data_array))
    filename, file_extension = os.path.splitext(audio)
    return_data = []
    for data in data_array:
        print(data)
        t1 = data['start'] * 1000 #Works in milliseconds
        t2 = data['stop'] * 1000
        newAudio = AudioSegment.from_wav(audio)
        newAudio = newAudio[t1:t2]
        newAudio.export(os.path.join(path,str(data['CurrentRegionID']+file_extension)), format="wav") #Exports to a wav file in the current path.
        try:
            clean_folder = os.path.join(path,'Clean')
            # log_folder = os.path.join(path,'Log')
            os.mkdir(clean_folder)    
            # os.mkdir(log_folder)
        except:
            pass
        
        
        
        predict(output_dir=clean_folder, input_file=os.path.join(path,str(data['CurrentRegionID'])+file_extension), 
            min_frequency=data['MinFrequencyValue'],freq_compression=data['FrequencyCompresion'],
            sr=data['Sr'],fft_size=data['FftSize'],n_freq_bins=data['FrequencyBins'],
            window_type=data['WindowType'], max_frequency=data['MaxFrequencyValue'],sequence_len=ceil(data['stop']-data['start']))
        return_data.append({
            'token':data['CurrentRegionID'],
            'audio_original':audio_name,
            'audio_cleaned':"denoised_0_"+str(data['CurrentRegionID'])+file_extension,
            'pdf_before':"net_input_spec_0_"+str(data['CurrentRegionID'])+".pdf",
            'pdf_after':"net_out_spec_0_"+str(data['CurrentRegionID'])+".pdf",
            'extension':file_extension
        })
        print("Audio recortado")
    return return_data

@csrf_exempt
@api_view(['GET'])
def playAudioFile(request):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    folder = 'posts'
    # MEDIA_DIR = os.path.join(BASE_DIR,'media')
    MEDIA_DIR = 'media'
    print(request.GET)
    elements = [MEDIA_DIR,folder, str(request.GET.get('token'))]
    # path = MEDIA_DIR + '\\' + folder + str(request.GET.get('token'))
    path = os.path.join(*elements)
    #path = MEDIA_DIR + '\\' + folder + str(request.GET.get('token'))
    check_file = os.path.join(path, str(request.GET.get('audio_name')))
    fname=check_file
    print(fname)
    # f = File(file =open(fname,"rb").read(), name = '6.wav')
    f = open(fname,"rb").read()
    response = HttpResponse()
    response.write(f)
    response['Content-Type'] ='audio/wav'
    response['Content-Length'] =os.path.getsize(fname)
    print(response)
    return response

@csrf_exempt
@api_view(['POST'])
def retrivePDF(request):
    # print("Request hecha")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    folder = 'posts'
    # MEDIA_DIR = os.path.join(BASE_DIR,'media')
    MEDIA_DIR = 'media'
    print(request.POST)
    # 'Hola',content_type='application/json; charset=utf-8'

    f = {}
    token = request.POST.get('tokens')
    pdf_names = request.POST.get('pdf_names').split(',')
    for fl in pdf_names:
      elements = [MEDIA_DIR, folder, str(token)]
      # path = MEDIA_DIR + '\\' + folder + str(token)
      path = os.path.join(*elements)
      last_elements = [path, 'Clean', str(fl)]
      # check_file = os.path.join(path, ('Clean\\'+ str(fl)))
      check_file = os.path.join(*last_elements)
      #path = MEDIA_DIR + '\\' + folder + str(token)
      #check_file = os.path.join(path, ('Clean\\'+ str(fl)))
      fname=check_file
    #   print("Fname",fname)
      # f = File(file =open(fname,"rb").read(), name = '6.wav')
      fs = FileSystemStorage()
      read = open(fname,"rb").read()
      f[fl]='/' + fname
    #   print('Value: ',fname)
    # print(f)
    response = HttpResponse(json.dumps(f))
    response['Access-Control-Allow-Origin']="*"
    
    response['Access-Control-Allow-Methods']= 'GET,PUT,POST,DELETE,PATCH,OPTIONS'
    
    # response['Content-Type'] ='application/pdf'
    
    # print(response)
    return response    

@api_view(['GET', 'POST'])
def store_json_data(request, format=None):
    """
    A view that can accept POST requests with JSON content.
    """
    # print(request.data)
    # value = parse.parse_qs(request.POST.get('name'))
    # for x in range(0,50000):

    #     if(x>0):
    #         print(x)
    # print(request.headers)
    # print(request)
    if request.method == 'GET':
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        folder = 'posts'
        # MEDIA_DIR = os.path.join(BASE_DIR,'media')
        MEDIA_DIR = 'media'
        print(request.GET)
        elements = [MEDIA_DIR ,folder, str(request.GET.get('token'))]
        path = os.path.join(*elements)
        #path = MEDIA_DIR + '\\' + folder + str(request.GET.get('token'))
        check_file = os.path.join(path, str(request.GET.get('audio_name')))
        with open(os.path.join(path,'data_json.json'),'r') as f:
            
            data = json.loads(f.read())
            
        return_data = split_audio(check_file,data, path,audio_name = str(request.GET.get('audio_name')))
        # print(return_data)
        return Response({'data': return_data},status=status.HTTP_200_OK)
    # elif request.method == 'POST' and request.POST.get['filetype'] == 'PDF':
    #     print("PDF")
    #     retrivePDF(request)
    # elif request.method == 'POST' and request.POST.get['filetype'] == 'Audio':
    #     playAudioFile(request)
    else:

        try:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            folder = 'posts'
            # MEDIA_DIR = os.path.join(BASE_DIR,'media')
            MEDIA_DIR = 'media'
            elements = [MEDIA_DIR,folder, str(request.POST.get('token'))]
            #path = MEDIA_DIR + '\\' + folder + str(request.POST.get('token'))
            path = os.path.join(*elements)
            check_file = os.path.join(path, str(request.POST.get('audio_name')))
            
            if (os.path.exists(check_file)):


                if os.path.exists(os.path.join(path, 'data_json.json')):
                    os.remove(os.path.join(path, 'data_json.json'))

                with open(os.path.join(path, 'data_json.json'), 'w') as file_json:
                    json.dump(literal_eval(request.POST.get('data')), file_json)
                
                print("Json con informacion creada")
                return Response({'data': request.POST.get('token'),'msg':'Json creado y guardado en back'},status=status.HTTP_200_OK)
            else:
                return Response('Proceso no se realizó desde el inicio porfavor volver a realizar el proceso de subir el audio',status=status.HTTP_404_NOT_FOUND)        
        except Exception as e:
            # print(e)
            return Response('Proceso no se  realizó desde el inicio porfavor volver a realizar el proceso de subir el audio',status=status.HTTP_404_NOT_FOUND)    
        # literal_eval(request.POST.get('data'))
    
class RoomView(generics.CreateAPIView):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer
class CreateRoomView(APIView):
    serializer_class = CreateRoomSerializer
    def post(self, request, format = None):
        if not self.request.session.exists(self.request.session.session_key):
            self.request.session.create()
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            guest_can_pause = serializer.data.get('guest_can_pause')
            votes_to_skip = serializer.data.get('votes_to_skip')
            host = self.request.session.session_key
            queryset = Room.objects.filter(host= host)
            if queryset.exists():
                room = queryset[0]
                room.guest_can_pause = guest_can_pause
                room.votes_to_skip = votes_to_skip
                room.save(update_fields = ['guest_can_pause','votes_to_skip'])
            else:
                room = Room(host = host, guest_can_pause = guest_can_pause, votes_to_skip = votes_to_skip)
                room.save()
            return Response(RoomSerializer(room).data, status=status.HTTP_201_CREATED)

class FileUploadView(APIView):
    # parser_classes = [MultiPartParser, FileUploadParser, FormParser]
    parser_classes = (MultiPartParser,)
    def get(self, request):

        return Response("GET API")

    def post(self, request, format=None):
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        folder = 'posts'
        # MEDIA_DIR = os.path.join(BASE_DIR,'media')
        MEDIA_DIR = 'media'
        
        
        try:
            FolderUser.objects.get(token = request.data['path'])
            folderModel = FolderUser.objects.get(token = request.data['path'])
        except FolderUser.DoesNotExist:
            # print("No existe carpeta")
            folderModel = FolderUser(token = request.data['path'])
            folderModel.save()
            try:
	            elements = [ MEDIA_DIR,folder, str(request.data['path'])]
	            os.mkdir(os.path.join(*elements))
                #os.mkdir(MEDIA_DIR + '\\' + folder + str(request.data['path']))    
            except:
                pass
        # print(SaveFile.objects.get(file = folder + str(request.data['path']) + '/' + str(request.data['file'])))
        
        try:
            
            
            SaveFile.objects.get(file = folder + str(request.data['path']) + '/' + str(request.data['file']))
            print("File save:",SaveFile.objects.get(file = folder + str(request.data['path']) + '/' + str(request.data['file'])))
            SaveFile.objects.get(file = folder + str(request.data['path']) + '/' + str(request.data['file'])).delete()
        except SaveFile.DoesNotExist:
            print("No existe archivo guardado")
            
        

        # print(folderModel)
        modelSaveFile = SaveFile(file = request.data['file'],path = folderModel)
        
        # print(request.data['file'])
        # serializer = SaveFileSerializer(modelSaveFile)
        
        # print("Serializer: ",serializer)
        
        # print(modelSaveFile.is_valid())
        if 'file' not in request.data:
            raise ParseError("Empty content")
            
        
        # if serializer.is_valid():
            
        if request.data['file'] == 'undefined':
            
            # serializer.save()
            
            return Response("No file selected", status=status.HTTP_400_BAD_REQUEST)
        else: 
            modelSaveFile.save()
            return Response("File uploaded", status=status.HTTP_200_OK)
            
        
        # ...
        # do some stuff with uploaded file
        # ...
# def handle_uploaded_file(f,name):
#     with open('controller\\media\\posts\\{}'.format(name), 'wb+') as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)


# def upload_file(request):
    
#     if request.method == 'POST':
#         print(request)
#         form = UploadFileForm(request.POST, request.FILES)
#         # print(form.check())
#         if form.is_valid():
#         # if form.check():
#             print("Entre")
#             handle_uploaded_file(request.FILES['file'],request.FILES['filename'].name)
#             # return Response("File uploaded", status=status.HTTP_200_OK)
#     else:
#         form = UploadFileForm()
    
