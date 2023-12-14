import cv2
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./inputs', help='path to source video')
parser.add_argument('--output', type=str, default='./output', help='output folder')
parser.add_argument('--rescale', type=bool, default=True, help='rescale or not')
parser.add_argument('--ratio', type=float, default=0.25, help='rescale ratio')
args = parser.parse_args()

def rescale_frame(Ap, ratio):
    heightAp, widthAp, channelsAp = Ap.shape
    heightAp, widthAp = int(heightAp*ratio), int(widthAp*ratio)
    Ap = cv2.resize(Ap, (widthAp, heightAp), interpolation=cv2.INTER_AREA)
    return Ap
    

frame = []
exts = ['*.png', '*.jpg', '*.jpeg']
for ext in exts:
   frame.extend(glob(os.path.join(args.input, ext)))
if len(frame) > 1:
    print('More than one frame file found. Please only provide one frame file.')
    exit()
elif len(frame) == 0:
    print('No frame file found. Please provide a frame file.')
    exit()
else:
    frame = frame[0]
    print('Frame file found: {}'.format(frame))

Ap = cv2.imread(frame)

if args.rescale:
    Ap = rescale_frame(Ap,args.ratio)
    
if not os.path.exists(args.output):
    os.makedirs(args.output)
cv2.imwrite(args.output+os.sep+'Ap.jpg',Ap)