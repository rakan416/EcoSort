# Generated by Django 5.0.6 on 2024-06-27 20:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('first', '0007_alter_user_options_alter_user_managers_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='date_joined',
            field=models.DateTimeField(auto_now=True),
        ),
    ]