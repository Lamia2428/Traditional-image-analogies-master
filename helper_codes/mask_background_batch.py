from glob import glob
from natsort import natsorted
import cv2
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input_frames', type=str, default='./output/B_frames', help='path to extracted frames')
parser.add_argument('--input_segmented', type=str, default='./output/B_segmented', help='path to segemented frames')
parser.add_argument('--output', type=str, default='./B', help='output folder')
args = parser.parse_args()

def mask_background_batch(input_frames,input_segmented,output_path):
    frames = natsorted(glob(input_frames+'/*.*'))
    images = [cv2.imread(img) for img in frames]

    segmented_frames = natsorted(glob(input_segmented+'/*.*'))
    segmented_images = [cv2.imread(img) for img in segmented_frames]

    masks = [np.where(np.any(img!=0, axis=2)) for img in segmented_images]

    for i,(Ap,A,mask) in enumerate(zip(images,segmented_images,masks)):
        Ap[mask] = A[mask]
        cv2.imwrite(output_path+os.sep+'B'+str(i)+'.jpg',Ap)

if not os.path.exists(args.output):
    os.makedirs(args.output)

mask_background_batch(args.input_frames,args.input_segmented,args.output)