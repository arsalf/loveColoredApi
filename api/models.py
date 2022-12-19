from django.db import models


class Species(models.Model):
    name = models.CharField(max_length=100)
    classification = models.CharField(max_length=100)
    language = models.CharField(max_length=100)


class Person(models.Model):
    name = models.CharField(max_length=100)
    birth_year = models.CharField(max_length=10)
    eye_color = models.CharField(max_length=10)
    species = models.ForeignKey(Species, on_delete=models.DO_NOTHING)


import matplotlib.pyplot as plt
import torch
from .module import colorizers

class Files(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='static/files/')

    # override the save method
    def save(self, *args, **kwargs):        
        # save dulu
        super(Files, self).save(*args, **kwargs)        

        # then modifiy the file
        # load colorizers
        colorizer_eccv16 = colorizers.eccv16(pretrained=True).eval()
        colorizer_siggraph17 = colorizers.siggraph17(pretrained=True).eval()
        # print(self.file)
        img = colorizers.load_img(str(self.file))
        (tens_l_orig, tens_l_rs) = colorizers.preprocess_img(img, HW=(256,256))

        print("processing image")
        img_bw = colorizers.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
        out_img_eccv16 = colorizers.postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = colorizers.postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # get the file name
        filename = self.file.name.split('.')[0]
                
        # save the file
        print("saving file")
        plt.imsave('%s_eccv16.png'%filename, out_img_eccv16)
        plt.imsave('%s_siggraph17.png'%filename, out_img_siggraph17)
