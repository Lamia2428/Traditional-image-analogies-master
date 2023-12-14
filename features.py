from sklearn.feature_extraction.image import extract_patches_2d as extract
import cv2
import numpy as np
from math import floor
from parameters import color


def get_features(img, causal=False, coarse=False):


    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgG = np.array(imgG,dtype=np.float32)/255.0

    height, width = imgG.shape  # dimensions of the current level of the gaussian pyramid
    
    padding = 1 if coarse else 2
    window = (3, 3) if coarse else (5, 5)

    dimentions_size = (window[0]*window[1]) if not causal else floor((window[0]*window[1])/2)
    features = np.zeros((height,width,dimentions_size))

    imgG_padded = cv2.copyMakeBorder(imgG,padding,padding,padding,padding,cv2.BORDER_DEFAULT)
    patchesG = extract(imgG_padded, window)[...,np.newaxis]
    
    if color:
        imgC = img
        imgC = np.array(imgC,dtype=np.float32)/255.0
        imgC_padded = cv2.copyMakeBorder(imgC,padding,padding,padding,padding,cv2.BORDER_DEFAULT)
        patchesC = extract(imgC_padded, window)
        patches = np.concatenate((patchesG, patchesC), axis=3)
    else:
        patches = patchesG

 # dimensions of the current level of the gaussian pyramid
    for i in range(height):
        for j in range(width):
                features[i, j, :] = patches[i * width + j].flatten()[0:features.shape[-1]]

    return features