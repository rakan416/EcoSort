# Generated by Django 5.0.6 on 2024-06-17 14:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('first', '0003_alter_post_banner'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='banner',
            field=models.ImageField(blank=True, default='default.jpg', upload_to='media/'),
        ),
    ]
