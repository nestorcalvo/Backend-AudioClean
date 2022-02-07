FROM python:3.9.6

ENV PYTHONUNBUFFERED 1

# Updating ubuntu and installing other necessary software
RUN apt-get update --yes \
    && apt-get install wget build-essential zlib1g-dev libncurses5-dev vim --yes

COPY . /audio_denoiser/
WORKDIR /audio_denoiser/

RUN pip install -r requirements.txt

EXPOSE 8000
CMD python manage.py makemigrations && python manage.py migrate && python manage.py runserver 0.0.0.0:8000