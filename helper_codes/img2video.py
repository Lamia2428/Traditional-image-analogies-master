import cv2
from glob import glob
from natsort import natsorted


files = natsorted(glob('out/*.*'))
images = [cv2.imread(img) for img in files]
height, width, layers = images[1].shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video1.mp4', fourcc, 30, (width, height))


for img in images:
    video.write(img)

video.release()