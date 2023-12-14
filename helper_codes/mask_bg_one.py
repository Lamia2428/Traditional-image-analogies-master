import cv2
import numpy as np
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_frames', type=str, default='./output/A_rescaled', help='path to extracted frames')
parser.add_argument('--input_segmented', type=str, default='./output/A_segmented', help='path to segemented frames')
parser.add_argument('--output', type=str, default='./', help='output folder')
args = parser.parse_args()


def add_bg(input_frame,input_segmented):
    seg = cv2.imread(input_frame)
    pic = cv2.imread(input_segmented)
    mask = np.where(np.any(pic!=0, axis=2))
    seg[mask] = pic[mask]
    return seg


frame = glob(args.input_frames+'/*.*')[0]
segmented = glob(args.input_segmented+'/*.*')[0]
bg = add_bg(frame,segmented)
cv2.imwrite(args.output+os.sep+"A.jpg",bg)