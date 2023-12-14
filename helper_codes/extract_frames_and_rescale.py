import cv2 
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./input', help='path to source video')
parser.add_argument('--output', type=str, default='./output', help='output folder')
parser.add_argument('--rescale', type=bool, default=True, help='rescale or not')
parser.add_argument('--ratio', type=float, default=0.25, help='rescale ratio')
args = parser.parse_args()

def extract_frames_and_rescale(video_path,output_path,rescale,ratio=0.25):
    def rescale_frame(frame_input, ratio):
        width = int(frame_input.shape[1] * ratio)
        height = int(frame_input.shape[0] * ratio)
        dim = (width, height)
        return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

    # Créer un objet de videoCapture
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    isExist = os.path.exists(output_path)
    if not isExist: 
        os.makedirs(output_path) 
    # Boucle pour extraire les frames
    i = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        # si on a récupérer la trame
        if rescale:
            frame = rescale_frame(frame,ratio)
        cv2.imwrite(output_path+os.sep+'B'+str(i)+'.jpg',frame)
        i+=1
    
    # libération 
    cap.release()

video = []
exts = ['*.mp4', '*.avi', '*.mov', '*.mpg', '*.mpeg', '*.mkv']
for ext in exts:
   video.extend(glob(os.path.join(args.input, ext)))
if len(video) > 1:
    print('More than one video file found. Please only provide one video file.')
    exit()
elif len(video) == 0:
    print('No video file found. Please provide a video file.')
    exit()
else:
    video = video[0]
    print('Video file found: {}'.format(video))

if not os.path.exists(args.output):
    os.makedirs(args.output)

output_path = os.path.join(args.output,'B_frames')
extract_frames_and_rescale(video, output_path, args.rescale, args.ratio)