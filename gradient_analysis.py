import sys
from optparse import OptionParser
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


# from eval import eval_net
sys.path.insert(0,'./utils/')
sys.path.insert(0,'./models/')
from models import * 
from utils import SequenceDataset
from utils import FaceDataset
import eval_metrics


# def main():
#     x = Variable(torch.randn(5, 5), requires_grad = True)
#     y = Variable(torch.randn(5, 5), requires_grad = True)
#     z = torch.sum(x + y)
#     z.backward()
#     print x.grad

def normalize(x):
    norm_of_x = torch.norm(x)
    return x/norm_of_x

def max_min_normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    
    return (x - x_min)/(x_max - x_min)


def get_valid_img(img, gradient, thresh = 0.2):
    valid_img =  np.zeros_like(img)
    valid_gradient = np.zeros_like(gradient)
    for m in range(gradient.shape[0]):
        for n in range(gradient.shape[1]):
            if gradient[m][n] > thresh:
                valid_img[m][n] = img[m][n]
                valid_gradient[m][n] = gradient[m][n]
    return valid_img,valid_gradient


def get_args():
    parser = OptionParser()

    parser.add_option('--batch_size', dest='batch_size', default=8,type='int', help='batch size')
    
    parser.add_option('--lr', dest='lr', default=0.0001, type='float', help='learning rate')
    
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    parser.add_option('--restore', action='store_true', dest='restore', default=False, help='use cuda')
    
    parser.add_option('--model_path',dest = 'model_path',default = '/home/',help = 'model save path')

    parser.add_option('--train_start_index',dest = 'train_start_index',default = 0,type = 'int',help = 'snapshot')

    parser.add_option('--loss',dest = 'loss',default='AddMarginLoss',help = 'loss type')

    parser.add_option('--dset_json_path',dest = 'dset_json_path',default = '/home',help = 'dataset json path')

    parser.add_option('--dset_hdf5_path',dest = 'dset_hdf5_path',default = '/home',help = 'dataset hdf5 path')

    parser.add_option('--gamma',dest = 'gamma',type = 'float', default = 0.2,help = 'lr decay')

    parser.add_option('--step_size',dest = 'step_size',type = 'float',default = 60000,help = 'step_size')

    parser.add_option('--max_iter',dest = 'max_iter',default = 100000,type = 'int',help = 'max_iter')

    parser.add_option('--load',dest = 'load',default = './checkpoints/',help = 'load checkpoints_dir')

    parser.add_option('--snapshot',dest = 'snapshot',default = 5000,type = 'float',help = 'snapshot')

    parser.add_option('--display',dest = 'display',default = 10,type = 'float',help = 'display')
    
    parser.add_option('--optim', dest = 'optim', default = 'SGD', help = 'optimizer type')

    parser.add_option('--num_classes',dest = 'num_classes', default = 888, type = 'int',help = 'num classes')    

    parser.add_option('--FRNet',dest = 'FRNet', default = 'resnet18',help = 'num classes')    

    parser.add_option('--in_ch', dest='in_ch', default=3, type='int', help='input_channel')

    parser.add_option('--input_size',dest = 'input_size',default = 256,type = 'int',help = 'input size')

    parser.add_option('--patch', action='store_true', dest='patch', default=False, help='patch analysis')

    (options, args) = parser.parse_args()

    return options

def eval_net(net = None, gpu = True, config = {}):
    #----------------------random choose two pictures-------------------#
    
    while 1:
        if config['patch']:
            img1,img2,filename1,filename2,label = config['eval_dset'].gradient_sample_for_patch()
            criterion = nn.MSELoss()
            label = torch.from_numpy(label).cuda()

            data1 = Variable(torch.from_numpy(img1).cuda(),requires_grad = True)
            output_cuda,pred1 = net(data1,label,mode = 'test',pred = True)
            norm_feature1 = normalize(output_cuda)

            data2 = Variable(torch.from_numpy(img2).cuda(),requires_grad = True)
            output_cuda,pred2 = net(data2,label,mode = 'test',pred = True)
            norm_feature2 = normalize(output_cuda)

            loss = criterion(norm_feature1, norm_feature2)

            loss.backward()

            print "loss: ",loss

            pred1 = np.squeeze(pred1.data.cpu().numpy())
            index1 = np.where(pred1 == np.max(pred1))

            pred2 = np.squeeze(pred2.data.cpu().numpy())
            index2 = np.where(pred2 == np.max(pred2))

            print "Ground Truth: ",label.cpu().numpy()
            print "Original Image pred: ",index1[0][0]
            print "Patch Image pred: ",index2[0][0]

            gradientofdata2 = max_min_normalize(np.abs((data2.grad).cpu().numpy()))
            print gradientofdata2.shape
        else:
            img1,img2,filename1,filename2 = config['eval_dset'].gradient_sample_for_ND()
            criterion = nn.MSELoss()

            data1 = Variable(torch.from_numpy(img1).cuda(),requires_grad = True)
            output_cuda = net(data1,mode = 'test')
            norm_feature1 = normalize(output_cuda)

            data2 = Variable(torch.from_numpy(img2).cuda(),requires_grad = True)
            output_cuda = net(data2,mode = 'test')
            norm_feature2 = normalize(output_cuda)

            loss = criterion(norm_feature1, norm_feature2)

            loss.backward()
            
            gradientofdata2 = max_min_normalize(np.abs((data2.grad).cpu().numpy()))

            valid_img2,valid_gradient2 = get_valid_img(np.squeeze(img2),np.squeeze(gradientofdata2), thresh = 0.1)

            print "loss: ",loss

            print gradientofdata2.shape
        
        if config['patch']:
            plt.subplot(131)
            plt.title("Original Img")
            plt.imshow(np.squeeze(img1))

            plt.subplot(132)
            plt.title("After adding patch")
            plt.imshow(np.squeeze(img2))
            
            plt.subplot(133)
            plt.title("Gradient Image")
            plt.imshow(np.squeeze(gradientofdata2))

            plt.suptitle(filename1)
            plt.show()
        else:
            plt.subplot(231)
            plt.title("Gradient Image")
            plt.imshow(np.squeeze(gradientofdata2))

            plt.subplot(232)
            plt.title("Original Img")
            plt.imshow(np.squeeze(img2))


            plt.subplot(233)
            plt.title("Valid Gradient")
            plt.imshow(valid_gradient2)

            plt.subplot(234)
            plt.title("Valid Image")
            plt.imshow(valid_img2)

            plt.subplot(235)
            plt.title("Image Difference")
            plt.imshow(np.squeeze(img1 - img2))


            plt.suptitle(filename1 + " " + filename2)
            plt.show()            




def main():
    args = get_args()
    
    TEST_DST = FaceDataset(json_path = args.dset_json_path, hdf5_path = args.dset_hdf5_path, in_ch = args.in_ch, \
    batch_size = args.batch_size, input_size = args.input_size)
    
    NET = FaceNet(in_ch = args.in_ch, FRNet = args.FRNet, metric = args.loss, num_classes = args.num_classes)

    NET.load_state_dict(torch.load(args.load))
    print('Model Restored from {}'.format(args.load))

    if args.gpu:
        NET.cuda()


    config = {}
    config['eval_dset'] = TEST_DST
    config['max_iter'] = args.max_iter
    config['snapshot'] = args.snapshot
    config['display'] = args.display
    config['lr'] = args.lr
    config['batch_size'] = args.batch_size
    config['step_size'] = args.step_size
    config['gamma'] = args.gamma
    config['model_path'] = args.model_path
    config['loss'] = args.loss
    config['train_start_index'] = args.train_start_index
    config['optim'] = args.optim
    config['input_size'] = args.input_size
    config['in_ch'] = args.in_ch
    config['net'] = NET
    config['patch'] = args.patch


    eval_net(net = NET, gpu = args.gpu, config = config)


if __name__ == "__main__":
    main()

    


