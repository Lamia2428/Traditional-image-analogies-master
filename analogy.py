
import numpy as np
import cv2
from features import get_features
from parameters import search_method
from search import nearest_neighbor_indexes, query_neighbors


def get_Af_and_indexes(Nlvl, A_L, Ap_L, coarse = False):
    A_f_list = []
    index_list = []
    for lvl in range(Nlvl, -1, -1):
        coarse = True if lvl == Nlvl else False

        A_f = get_features(A_L[lvl], coarse=coarse)
        Ap_f = get_features(Ap_L[lvl], causal=True, coarse=coarse)

        A_f = np.concatenate((A_f, Ap_f), 2)
        # initialize additional feature sets and B mats
        if lvl < Nlvl:
            Ad = cv2.resize(A_L[lvl+1], (A_L[lvl].shape[1],A_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
            Ad_f = get_features(Ad, coarse=coarse)
            
            Apd = cv2.resize(Ap_L[lvl + 1], (Ap_L[lvl].shape[1], Ap_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
            Apd_f = get_features(Apd, coarse=coarse)

            A_f = np.concatenate((A_f, Ad_f, Apd_f), 2)

        # put feature list into index format M*N,numFeatures (25+12)
        source_f_vect = A_f.reshape(-1, A_f.shape[-1])

        """  Begin Neighbor Search Methods  """
        print(f"Building {search_method} index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        index,flann = nearest_neighbor_indexes(source_f_vect)
        print(f"{search_method} index done...")
        """  End Neighbor Search Methods  """
        A_f_list.insert(0,A_f)
        index_list.insert(0,(index,flann))

    return index_list, A_f_list


def make_analogy(index_pyf,A_f,lvl, Nlvl, Ap_L, B_L, Bp_L, s_L, kappa=0):

    coarse = True if lvl == Nlvl else False

    index,flann = index_pyf

    # initialize mat by taking previous pyramid level and resize it to the same shape as the current level
    # for lvl=Nlvl you can initialize it with current Ap or with some randomization function
    if lvl < Nlvl:
        Bp_L[lvl] = cv2.resize(Bp_L[lvl+1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        Bp_L[lvl] = cv2.resize(B_L[lvl], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)

    B_f = get_features(B_L[lvl], coarse=coarse)
    Bp_f = get_features(Bp_L[lvl], causal=True, coarse=coarse)
    B_f = np.concatenate((B_f, Bp_f), 2)
    if lvl < Nlvl:
        B_up = cv2.resize(B_L[lvl + 1], dsize=(B_L[lvl].shape[1], B_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Bp_up = cv2.resize(Bp_L[lvl + 1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        B_up_f = get_features(B_up, coarse=coarse)
        Bp_up_f = get_features(Bp_up, coarse=coarse)
        B_f = np.concatenate((B_f, B_up_f, Bp_up_f), 2)


    target_f_vect = B_f.reshape(-1, B_f.shape[-1])
    print("target_f_vect shape", target_f_vect.shape)

    # query neighbors
    neighbors , distances = query_neighbors(index,target_f_vect,flann=flann)

    coh_chosen = 0
    # coh_fact is squared to get it closer to the performance as described in Hertzmann paper
    coh_fact = (1.0 + (2.0**(lvl - Nlvl)) * kappa)**2

    for x in range(Bp_L[lvl].shape[0]):
        if x%25 == 0:
            print("Rastering row", x, "of",Bp_L[lvl].shape[0])
        for y in range(Bp_L[lvl].shape[1]):
            
            position = np.ravel_multi_index((x,y), (Bp_L[lvl].shape[0], Bp_L[lvl].shape[1]))

            neighbor_app,distance = neighbors[position],distances[position]
            distance_app = distance**2
            m,n = np.unravel_index(neighbor_app, (A_f.shape[0], A_f.shape[1]))

            if kappa > 0:
                neighbor_coh, distance_coh = get_coherent(A_f, target_f_vect[position], x, y, s_L[lvl], coarse=coarse)
 
                got_coh = (neighbor_coh != [-1, -1])
                if got_coh and distance_coh <= distance_app * coh_fact:
                    m,n = neighbor_coh
                    coh_chosen += 1

            Bp_L[lvl][x,y,:] = Ap_L[lvl][m,n,:]  # move value into Bprime

            # save s
            s_L[lvl][x, y, 0] = m
            s_L[lvl][x, y, 1] = n

    print("Coherent pixel chosen", coh_chosen, "/", Bp_L[lvl].size, "times.")
    return Bp_L[lvl]


def get_coherent(A_f,B_f,q_x,q_y,s,coarse = False):  # tuned for 5x5 patches only
    min_distance = np.inf
    cohxy = [-1, -1]

    start = -1 if coarse else -2
    stop = 2 if coarse else 3

    for i in range(start, stop, 1):
        for j in range(start, stop, 1):

            r_i,r_j = q_x+i,q_y+j
            if i == 0 and j == 0:  # only do causal portion
                break
            if r_i >= s.shape[0] or r_j >= s.shape[1]:
                continue

            sx,sy = s[r_i,r_j]
            if sx == -1 or sy == -1:
                continue
            #q_x - r_i = -i
            rx, ry = sx-i, sy-j

            if rx < 0 or rx >= A_f.shape[0] or ry < 0 or ry >= A_f.shape[1]:
                continue

            rstar = np.sum((A_f[rx,ry,:]-B_f)**2)

            if rstar < min_distance:
                min_distance = rstar
                cohxy = rx, ry

    return cohxy, min_distance

