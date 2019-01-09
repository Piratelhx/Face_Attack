import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import json


class SequenceDataset():

    def __init__(self,file_path):

        self.file = h5py.File(file_path + '/train_ND_2006.hdf5')
        self.anno = json.load(open(file_path + '/new_label.json'))
        random.shuffle(self.anno)
        self.idx = 0
        self.ch,self.h,self.w = self.file[self.anno[0]['filename']+'.txt'].shape


    def next_batch(self,batch = 64, input_size = None):

        if input_size:
            data = np.zeros((batch,self.ch,input_size[0],input_size[1]),dtype = np.float32)
            pad_h = (input_size[0] - self.h) / 2
            pad_w = (input_size[1] - self.w) / 2
        else:
            data = np.zeros((batch,self.ch,self.h,self.w),dtype = np.float32)
        label = np.zeros(batch,dtype = np.int64)
        for i in range(batch):

            if self.idx == len(self.anno) - 1:
                self.idx = 0

            file_name = self.anno[self.idx]['filename']+'.txt'
            if input_size:
                data[i,:,pad_h:input_size[0] - pad_h,pad_w:input_size[1] - pad_w] = self.file[file_name]
                label[i] = self.anno[self.idx]['label']
            else:    
                data[i] = self.file[file_name]
                label[i] = self.anno[self.idx]['label']
            self.idx += 1
        buff = {}
        buff['data'] = data
        buff['label'] = label
        return buff


class SequenceDataset2():

    def __init__(self,file_path):

        self.file = h5py.File(file_path)
        self.__gen_label()
        self.idx = 0

    def __gen_label(self):

        self.file_list = self.file.keys()
        self.ch,self.h,self.w = self.file[self.file_list[0]].shape

        self.label = {}
        keys = []
        for each in self.file_list:
            key = each.split('_')[1].split('d')[0]
            if key in keys:
                self.label[each] = keys.index(key)
            else:
                keys.append(key)
                self.label[each] = keys.index(key)

        random.shuffle(self.file_list)

    def next_batch(self,batch = 64, input_size = None):

        if input_size:
            data = np.zeros((batch,self.ch,input_size[0],input_size[1]),dtype = np.float32)
            pad_h = (input_size[0] - self.h) / 2
            pad_w = (input_size[1] - self.w) / 2
        else:
            data = np.zeros((batch,self.ch,self.h,self.w),dtype = np.float32)
        label = np.zeros(batch,dtype = np.int64)
        for i in range(batch):

            if self.idx == len(self.file_list) - 1:
                self.idx = 0

            file_name = self.file_list[self.idx]
            if input_size:
                data[i,:,pad_h:input_size[0] - pad_h,pad_w:input_size[1] - pad_w] = self.file[file_name]
                label[i] = self.label[file_name]
            else:    
                data[i] = self.file[file_name]
                label[i] = self.label[file_name]
            self.idx += 1
        buff = {}
        buff['data'] = data
        buff['label'] = label
        return buff
        



def main():

    dataset = SequenceDataset('/media/tanyang/xiaozl/synthesis_data/sequency_hdf5_LQ')

    for i in range(2):
        buff= dataset.next_batch(batch = 20, input_size = (128,128))

        print buff['label']
        print buff['data'].shape
        for i in range(5):
            plt.imshow(buff['data'][0,i])
            plt.show()


if __name__ == '__main__':
    main()