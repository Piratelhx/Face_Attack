import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import math
sys.path.insert(0,'./utils/')
sys.path.insert(0,'./models/')
from models import * 
from utils import SequenceDataset
from utils import FaceDataset
import eval_functions

TEST_DATASET_H5PY_DIR = "/home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.hdf5"
TEST_DATASET_GALLERY_JSON_DIR = "/home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.json"
TEST_DATASET_PROBE_NEUTRAL_JSON_DIR = "/home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.json"
TEST_DATASET_FRONTIAL_JSON_DIR = "/home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.json"


def normalize(x):
    
	mod = np.linalg.norm(x)

	return x / mod

def eval_data_BOS(config):  
    print "---------------------------Begin Testing---------------------------"
    GALLERY_DST = FaceDataset(json_path = TEST_DATASET_GALLERY_JSON_DIR, hdf5_path = TEST_DATASET_H5PY_DIR, in_ch = config['in_ch'], \
    batch_size = config['batch_size'], input_size = config['input_size'], mode = "test")

    PROBE_NEUTRAL_DST = FaceDataset(json_path = TEST_DATASET_PROBE_NEUTRAL_JSON_DIR, hdf5_path = TEST_DATASET_H5PY_DIR, in_ch = config['in_ch'], \
    batch_size = config['batch_size'], input_size = config['input_size'], mode = "test")

    FRONTIAL_DST = FaceDataset(json_path = TEST_DATASET_FRONTIAL_JSON_DIR, hdf5_path = TEST_DATASET_H5PY_DIR, in_ch = config['in_ch'], \
    batch_size = config['batch_size'], input_size = config['input_size'], mode = "test")


    net = config['net']
    
    probe_feature = {}
    probe_neutral_feature = {}
    gallery_feature = {}
    nonneutral_feature = {}

    while GALLERY_DST.has_next_batch():
        
        buff = GALLERY_DST.get_batch_samples()
        data = buff["data"]
        label = buff['label']
        isvalid = buff["flag"]

        data = torch.from_numpy(data).cuda()

        with torch.no_grad():
            for k in range(isvalid.shape[0]):  
                if isvalid[k] ==  1:       
                    input_data = torch.from_numpy(np.expand_dims(data[k],axis = 0)).cuda()
                    output_cuda = net(input_data,mode = 'test')
                    feature = output_cuda.cpu().numpy()
                    norm_feature = normalize(feature)
                    if  gallery_feature.has_key(label[k]):
                        gallery_feature[label[k]].append(norm_feature.copy())
                    else:
                        gallery_feature[label[k]] = []
                        gallery_feature[label[k]].append(norm_feature.copy())
    


    while PROBE_NEUTRAL_DST.has_next_batch():

        buff = PROBE_NEUTRAL_DST.get_batch_samples()
        data = buff["data"]
        label = buff['label']
        isvalid = buff["flag"]

        data = torch.from_numpy(data).cuda()

        with torch.no_grad():
            for k in range(isvalid.shape[0]):  
                if isvalid[k] ==  1:  
                    input_data = torch.from_numpy(np.expand_dims(data[k],axis = 0)).cuda()
                    output_cuda = net(input_data,mode = 'test')
                    feature = output_cuda.cpu().numpy()
                    norm_feature = normalize(feature)

                    if  probe_neutral_feature.has_key(label[k]):
                        probe_neutral_feature[label[k]].append(norm_feature.copy())
                    else:
                        probe_neutral_feature[label[k]] = []
                        probe_neutral_feature[label[k]].append(norm_feature.copy())
                    
                    if probe_feature.has_key(label[k]):
                        probe_feature[label[k]].append(norm_feature.copy())
                    else:
                        probe_feature[label[k]] = []
                        probe_feature[label[k]].append(norm_feature.copy())
            

    while FRONTIAL_DST.has_next_batch():
        buff = FRONTIAL_DST.get_batch_samples()
        data = buff["data"]
        label = buff['label']
        isvalid = buff["flag"]

        data = torch.from_numpy(data).cuda()

        with torch.no_grad():
            for k in range(isvalid.shape[0]):  
                if isvalid[k] ==  1:  
                    input_data = torch.from_numpy(np.expand_dims(data[k],axis = 0)).cuda()
                    output_cuda = net(input_data,mode = 'test')
                    feature = output_cuda.cpu().numpy()
                    norm_feature = normalize(feature)
                    
                    if probe_feature.has_key(label[k]):
                        probe_feature[label[k]].append(norm_feature.copy())
                    else:
                        probe_feature[label[k]] = []
                        probe_feature[label[k]].append(norm_feature.copy())   

                    if nonneutral_feature.has_key(label[k]):
                        nonneutral_feature[label[k]].append(norm_feature.copy())
                    else:
                        nonneutral_feature[label[k]] = []
                        nonneutral_feature[label[k]].append(norm_feature.copy())

    

    False_acceptance_list = [0.001]   
    for i in range(len(False_acceptance_list)):
        # print "----------------The Following is neutral vs neutral False Acceptance result---------------" 
        neutral_vs_neutral = eval_functions.test_False_acceptance_Bos(gallery_feature,probe_neutral_feature,False_acceptance = False_acceptance_list[i])
        
        # print "----------------The Following is neutral vs nonneutral False Acceptance result---------------" 
        neutral_vs_noneneutral = eval_functions.test_False_acceptance_Bos(gallery_feature,nonneutral_feature,False_acceptance = False_acceptance_list[i])
        
        # print "----------------The Following is neutral vs All False Acceptance result---------------" 
        neutral_vs_all = eval_functions.test_False_acceptance_Bos(gallery_feature,probe_feature,False_acceptance = False_acceptance_list[i])

        return_value = neutral_vs_all
        
        print "FAR result under {} False Acceptance is {} {} {}".format(False_acceptance_list[i],neutral_vs_neutral,neutral_vs_noneneutral,neutral_vs_all) 


    neutral_vs_neutral = eval_functions.test_cmc(gallery_feature,probe_neutral_feature,top_k = 1)    
    neutral_vs_noneneutral = eval_functions.test_cmc(gallery_feature,nonneutral_feature,top_k = 1)    
    neutral_vs_all = eval_functions.test_cmc(gallery_feature,probe_feature,top_k = 1,imshow_cmc = False)  
 
    print "TOP 0 is {} {} {}".format(neutral_vs_neutral[0],neutral_vs_noneneutral[0],neutral_vs_all[0]) 
    # print "----------------The Following is neutral vs neutral CMC result---------------" 
    # for i in range(len(neutral_vs_neutral)):
    #     print "Top ",str(i),"is: ",neutral_vs_neutral[i]

    # print "----------------The Following is neutral vs nonneutral CMC result---------------" 
    # for i in range(len(neutral_vs_nonneutral)):
    #     print "Top ",str(i),"is: ",neutral_vs_nonneutral[i]

    # print "----------------The Following is neutral vs all CMC result---------------"  
    # for i in range(len(neutral_vs_all)):
    #     print "Top ",str(i),"is: ",neutral_vs_all[i]
    
    print "---------------------------End Testing---------------------------"
    return return_value


