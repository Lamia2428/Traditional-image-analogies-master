import numpy as np
import cv2
from parameters import *
from analogy import make_analogy, get_Af_and_indexes
import os
from glob import glob
from natsort import natsorted

def do_analogy(input, output):

    def get_pyramid(image, levels):
        img = image.copy()
        pyr = [img]
        for i in range(levels):
            img = cv2.pyrDown(img)
            pyr.append(img)

        return pyr


    A_path = glob(os.path.join(input, 'A.*'))[0]
    Ap_path = glob(os.path.join(input, 'Ap.*'))[0]
    imgA = cv2.imread(A_path)
    imgAp = cv2.imread(Ap_path)
    A_L = get_pyramid(imgA, pyr_levels)
    Ap_L =get_pyramid(imgAp, pyr_levels)

    #get path to all B images
    B_paths = natsorted(glob(os.path.join(input, "B", 'B*.*')))

    print(B_paths)
    #load all B images
    listB = [cv2.imread(path) for path in B_paths]
    #for each B image make a pyramid
    B_L_list = [get_pyramid(imgB, pyr_levels) for imgB in listB]



    Bp_L_list = []
    s_list = []
    # for each B image make Bp and s at each level
    for B_L in B_L_list:
        Bp_L = []
        s = []
        for i in range(pyr_levels+1):
            Bp_L.append(np.zeros(B_L[i].shape))
            s.append(np.full((B_L[i].shape[0],B_L[i].shape[1],2),-1))
        Bp_L_list.append(Bp_L)
        s_list.append(s)


    index_list, A_f_list = get_Af_and_indexes(pyr_levels, A_L, Ap_L)

    # for each B image make analogy
    for i, (B_L, Bp_L, s) in enumerate(zip(B_L_list, Bp_L_list, s_list)):
        print("Starting Image: ", i, "of ", len(B_L_list))
        for lvl in range(pyr_levels, -1, -1):
            
            index_pyf= index_list[lvl]
            A_f = A_f_list[lvl]

            print("Starting Level: ", lvl, "of ", pyr_levels)
            Bp_L[lvl] = make_analogy(index_pyf,A_f,lvl, pyr_levels, Ap_L, B_L, Bp_L, s, kappa)

        imgBp = Bp_L[0]
        write_name = 'Bp'+str(i)+'.jpg'
        cv2.imwrite(os.path.join(output, write_name),imgBp)
