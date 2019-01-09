#coding=utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import random
from optparse import OptionParser
import math
import cPickle as pickle

import json
import h5py

# def cos_distance(x1, x2):
#     return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cos_distance(a,b):
    len_a = np.sqrt(np.sum(np.square(a)))
    len_b = np.sqrt(np.sum(np.square(b)))
    dist = np.sum(np.multiply(a,b))/(len_a*len_b)
    return dist


def test_False_acceptance(correspond_label_features,False_acceptance = 0.001):
    try:
        isinstance(correspond_label_features,dict)
    except:
        print "You should input a feature in dict form like this: Feature['label'] = list(a set of numpy data)"
    else:
        #----------------------Diff Person Dist--------------------------#
        diff_person = []
        for key in correspond_label_features.keys():  
            diff_person.append(correspond_label_features[key][0])
        diff_person_dist = []
        for i in range(0,len(diff_person)):
            for j in range(i+1,len(diff_person)):
                dist =cos_distance(diff_person[i],diff_person[j])
                diff_person_dist.append(dist)

        #----------------------Same Person Dist--------------------------#
        same_person_dist = []
        for key in correspond_label_features.keys():
            for i in range(len(correspond_label_features[key])):
                for j in range(i+1,len(correspond_label_features[key])): 
                    dist = cos_distance(correspond_label_features[key][i],correspond_label_features[key][j])
                    same_person_dist.append(dist)

        diff_person_dist = sorted(diff_person_dist)
        num = int(float(len(diff_person_dist))*(1 - False_acceptance))
        thresh = diff_person_dist[num]
        same_right_num = 0
        for j in range(len(same_person_dist)):
            if same_person_dist[j]>thresh:
                same_right_num+=1
        
        print "max diff_person: ",max(diff_person_dist)   
        print "min diff_person: ",min(diff_person_dist)   
        print "mean diff_person: ",np.mean(diff_person_dist)

        print "max same_person: ",max(same_person_dist)   
        print "min same_person: ",min(same_person_dist) 
        print "mean same_person: ",np.mean(same_person_dist) 

        print "len(diff_person_dist):",len(diff_person_dist)
        print "Thresh: ",num," ",diff_person_dist[num]

        print "same_right_num: ",same_right_num
        print "len of same_person_list: ",len(same_person_dist)
        print "Under ",False_acceptance," Acceptance: ","the same person acceptance accuracy is ",float(same_right_num)/len(same_person_dist)

def test_False_acceptance_Bos(gallery_features,probe_features,False_acceptance = 0.001):
    try:
        isinstance(gallery_features,dict) and isinstance(probe_features,dict)
    except:
        print "You should input a feature in dict form like this: Feature['label'] = list(a set of numpy data)"
    else:
        #----------------------Diff Person Dist and Same Person Dist--------------------------#
        diff_person_dist = []
        same_person_dist = []       
        for gallery_key in gallery_features.keys():
            for probe_key in probe_features.keys():
                for i in range(len(probe_features[probe_key])):
                    dist = cos_distance(gallery_features[gallery_key][0],probe_features[probe_key][i])
                    if probe_key == gallery_key:
                        same_person_dist.append(dist.copy())
                    else:
                        diff_person_dist.append(dist.copy())

        diff_person_dist = sorted(diff_person_dist)
        num = int(float(len(diff_person_dist))*(1 - False_acceptance))
        thresh = diff_person_dist[num]
        same_right_num = 0
        for j in range(len(same_person_dist)):
            if same_person_dist[j]>thresh:
                same_right_num+=1
        
        # print "max diff_person: ",max(diff_person_dist)   
        # print "min diff_person: ",min(diff_person_dist)   
        # print "mean diff_person: ",np.mean(diff_person_dist)

        # print "max same_person: ",max(same_person_dist)   
        # print "min same_person: ",min(same_person_dist) 
        # print "mean same_person: ",np.mean(same_person_dist) 

        # print "len(diff_person_dist):",len(diff_person_dist)
        # print "Thresh: ",num," ",diff_person_dist[num]

        # print "same_right_num: ",same_right_num
        # print "len of same_person_list: ",len(same_person_dist)
        # print "Under ",False_acceptance," Acceptance: ","the same person acceptance accuracy is ",float(same_right_num)/len(same_person_dist)
        return float(same_right_num)/len(same_person_dist)

def test_cmc(correspond_label_gallery_features,correspond_label_probe_features,top_k = 1,imshow_cmc = False):
    try:
        isinstance(correspond_label_gallery_features,dict) and \
        isinstance(correspond_label_probe_features,dict)
    except:
         print "You should input a feature in dict form like this: Feature['label'] = list(a set of numpy data)"
    else:
        # print "length of gallery: ",len(correspond_label_gallery_features)
        # print "length of test: ",len(correspond_label_probe_features)
        top = {}
        for i in range(top_k):
            top[i] = 0
        
        # print "Calculating  CMC ..."
        test_num = 0
        for key in correspond_label_probe_features.keys():
            for i in range(len(correspond_label_probe_features[key])):
                # print "CMC Testing: ",test_num
                test_num += 1
                res = {}
                for gallery_key in correspond_label_gallery_features.keys():
                    dist =cos_distance(correspond_label_probe_features[key][i],correspond_label_gallery_features[gallery_key][0])
                    res[gallery_key] = dist.copy()
                
                score = sorted(res.items(),key = lambda item:-item[1])
                for z in range(top_k):
                    for k in range(z+1):
                        if score[k][0] == key:
                            top[z]+=1
                            break


        final = []
        for i in range(top_k):
            final.append(float(top[i])/test_num)
            # print "Top ",str(i+1)," is: ",float(top[i])/test_num

        if imshow_cmc:
            plt.figure()  
            x = np.arange(0,top_k)  
            plt.plot(x,final,color="red",linewidth=2)
            plt.xlabel("Rank")  
            plt.ylabel("Matching Rate")
            plt.xticks(np.arange(1,top_k + 1,1)) 
            plt.title("CMC Curve")
            plt.show()
        return final


    



