# Generated by Django 5.0.6 on 2024-06-28 00:03

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('first', '0009_upimg_detect_img_upimg_hasil_upimg_tanggal_detect_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upimg',
            name='tanggal_detect',
            field=models.DateTimeField(default=datetime.datetime(2024, 6, 28, 0, 2, 45, 867498, tzinfo=datetime.timezone.utc)),
        ),
    ]
