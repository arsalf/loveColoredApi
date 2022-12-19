
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import cv2
import matplotlib.pyplot as plt
from .eccv16 import *
from .siggraph17 import *
import os
import glob
import re

# Function to extract frames
def colorfullPerFrame(path):
    # load the model    
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
 
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    vidName = path.split('/')[-1].split('.')[0]        

    # check if folder exist
    if not os.path.exists('static/video/'+vidName):
        os.makedirs('static/video/'+vidName)

    # Used as counter variable
    count = 0
 
    # checks whether frames were extracted
    success = 1
 
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if(not success):
            break
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.asarray(Image.fromarray(img))
        if(img.ndim==2):
            img = np.tile(img[:,:,None],3)
        
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

        print("processing image frame...")    
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())        
        # Saves the frames with frame-count        
        print("saving frame colorfull...")
        plt.imsave('static/video/'+vidName+"/frame%d.png" % count, out_img_siggraph17)

        count += 1

def grayfullPerFrame(path): 
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    vidName = path.split('/')[-1].split('.')[0]        

    # check if folder exist
    if not os.path.exists('static/video/'+vidName):
        os.makedirs('static/video/'+vidName)

    # Used as counter variable
    count = 0
 
    # checks whether frames were extracted
    success = 1
 
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if(not success):
            break

        # Converts to grayscale
        print("processing image frame...")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Saves the frames with frame-count
        print("saving frame grayfull...")
        cv2.imwrite('static/video/'+vidName+"/frame%d.jpg" % count, img)

        count += 1

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def resize_img(img, HW=(256,256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def saveImgColorfull(file):
    # load the model
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    # print(file)
    img = load_img(str(file))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

    print("processing image...")
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    # get the file name
    filename = file.name.split('.')[0]
    # save the file
    print("saving file...")
    plt.imsave('%s_bw.png'%filename, img_bw)
    plt.imsave('%s_eccv16.png'%filename, out_img_eccv16)
    plt.imsave('%s_siggraph17.png'%filename, out_img_siggraph17)

def saveVideoColorfull(file):
    # colorfull per frame
    colorfullPerFrame(file.name)

    # convert to video
    vidName = file.name.split('/')[-1].split('.')[0]
    vidFolder = file.name.split('/')[-2]
    img_array = []
    print("menyatukan frame:")
    for filename in  sorted(glob.glob("static/video/"+vidName+'/*.png'), key=lambda s: int(re.search(r'frame(\d+)', s).groups()[0])):
        img = cv2.imread(filename)
        print(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter("static/video/"+vidName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def saveImgGrayfull(file):
    # load img with cv2
    img = cv2.imread(str(file))
    # convert to grayscale
    print("processing image...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # save the grayscale image
    print("saving file...")
    cv2.imwrite(file.name.split('.')[0]+'_grayfull.jpg', gray)

def saveVideoGrayfull(file):
    # colorfull per frame
    grayfullPerFrame(file.name)

    # convert to video
    vidName = file.name.split('/')[-1].split('.')[0]    
    img_array = []
    print("menyatukan frame:")
    # sorted for the frame order
    
    for filename in sorted(glob.glob("static/video/"+vidName+'/*.jpg'), key=lambda s: int(re.search(r'frame(\d+)', s).groups()[0])):
        img = cv2.imread(filename)
        print(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter("static/video/"+vidName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    