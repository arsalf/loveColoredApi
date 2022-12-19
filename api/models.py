from django.db import models

import matplotlib.pyplot as plt
import torch
from .module import colorizers

class Files(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='static/')

    # override the save method
    def save(self, *args, **kwargs):        
        # if image
        if self.file.name.endswith('.jpg') or self.file.name.endswith('.png'):
            # upload to static/files/image            
            self.file.name = 'image/' + self.file.name
            self.file.upload_to = 'static/image/'
            super(Files, self).save(*args, **kwargs)        

            # then modifiy the file    
            colorizers.saveImgColorfull(self.file)
        # if video
        elif self.file.name.endswith('.mp4') or self.file.name.endswith('.avi'):
            # upload to static/files/video
            self.file.name = 'video/' + self.file.name
            self.file.upload_to = 'static/video/'
            super(Files, self).save(*args, **kwargs)  

            # then modifiy the file
            colorizers.saveVideoColorfull(self.file)
        # else then raise error
        else:
            raise Exception('File type not supported')