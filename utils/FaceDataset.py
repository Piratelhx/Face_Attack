import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import cv2

def add_patch(img, max_patch_num = 6,max_patch_size = 20):
    random_point_num = np.random.randint(max_patch_num)
    # plt.subplot(121)
    # plt.imshow(img.copy())
    for i in range(random_point_num):
        patch_size = np.random.randint(0,max_patch_size)
        tmp = (patch_size - 1)/2
        random_point_row = np.random.randint(tmp,img.shape[0]-tmp)
        random_point_col = np.random.randint(tmp,img.shape[1]-tmp)
        for m in range(-tmp,tmp):
            for n in range(-tmp,tmp):
                img[random_point_row+m,random_point_col+n] = 0
    
    return img

class FaceDataset():

    def __init__(self, json_path, hdf5_path, in_ch = 3, batch_size = 64, input_size = 256, mode = "train"):
        self.file = h5py.File(hdf5_path,'r')
        self.anno = json.load(open(json_path,'r'))
        # print "loading hdf5 data: {} and json data {} ok".format(hdf5_path, json_path)
        self.idx = 0
        self.in_ch = in_ch
        self.input_size = input_size
        self.batch_size = batch_size
        self.length = len(self.anno)

        

        if mode == "train":
            random.shuffle(self.anno)
            print "TRAIN NUM: {}".format(self.length)
        
        if mode == "test":
            pass
        

    def next_batch(self, data_augmentation = False, max_rotatation_num = 9):
        
        data = np.zeros((self.batch_size, self.in_ch, self.input_size, self.input_size), dtype = np.float32)
        label = np.zeros(self.batch_size, dtype = np.int64)

        for i in range(self.batch_size):
            if self.idx == len(self.anno) - 1:
                random.shuffle(self.anno)
                self.idx = 0
           
            filename = self.anno[self.idx]['filename']
            img_index = random.randint(0,max_rotatation_num)
            img = np.asarray(self.file[filename][img_index]) 
           
            if data_augmentation: 
                data[i] = add_patch(img)
            else:
                for ch in range(self.in_ch):
                    data[i,ch,:] = img
            
            label[i] = self.anno[self.idx]['label']
            self.idx += 1
        buff = {}
        buff['data'] = data
        buff['label'] = label

        return buff
    
    def get_batch_samples(self):  
        label = np.zeros(self.batch_size, dtype = np.int64)
        data = np.zeros((self.batch_size, self.in_ch, self.input_size, self.input_size), dtype = np.float32)
        flag = np.ones(self.batch_size,dtype = np.int32)
        buff = {}
        for i in range(self.batch_size):
            if self.idx < self.length:
                filename = self.anno[self.idx]['filename']
                img = np.asarray(self.file[filename][0]) 
                data[i] = img
                label[i] = self.anno[self.idx]['label']
                self.idx += 1
            else:
                flag[i] = 0
            

        buff["data"] = data
        buff["label"] = label
        buff["flag"] = flag
        return buff
    
    def has_next_batch(self):
        return self.idx < self.length
    
    def return_id(self):
        return self.idx

    def return_length(self):
        return self.length
