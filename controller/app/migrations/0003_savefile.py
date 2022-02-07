# Generated by Django 3.2.7 on 2021-09-26 05:16

import app.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_alter_room_code'),
    ]

    operations = [
        migrations.CreateModel(
            name='SaveFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.FileField(blank=True, db_index=True, upload_to=app.models.user_directory_path)),
            ],
        ),
    ]
