# Generated by Django 3.2.7 on 2021-09-27 06:09

import app.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_auto_20210927_0106'),
    ]

    operations = [
        migrations.AlterField(
            model_name='savefile',
            name='file',
            field=models.FileField(upload_to=app.models.user_directory_path),
        ),
    ]
