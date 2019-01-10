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

# from eval import eval_net
sys.path.insert(0,'./utils/')
sys.path.insert(0,'./models/')
from models import * 
from utils import SequenceDataset
from utils import FaceDataset
import eval_metrics
import matplotlib.pyplot as plt



def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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

    parser.add_option('--test_iter', dest='test_iter', default=8,type='int', help='test iter')

    (options, args) = parser.parse_args()

    
    return options

def train_net(net, gpu=False, config={}, LOG_FOUT = None):
    
    try:

        RES = 0
        BEST_RES = 0

        FRDATASET = config['train_dset']

        print('Starting training...')
        
        if config['optim'] == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0005)
        elif config['optim'] == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = 0.00005)    


        criterion = nn.CrossEntropyLoss()

        loss_count = 0
        time_start = time.time()

        for iter_ in range(config['train_start_index'], config['max_iter']):
            
            buff = FRDATASET.next_batch(data_augmentation = True)

            data = buff['data']
            label = buff['label']
            if gpu:
                data = torch.from_numpy(data).cuda()
                label = torch.from_numpy(label).cuda()

            net_pred = net(data,label)
            loss = criterion(net_pred, label)
            loss_count += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter_ + 1) % config['snapshot'] == 0:
                RES = eval_metrics.eval_data_BOS(config)

                if RES > BEST_RES:
                    BEST_RES = RES
                    torch.save(net.state_dict(), os.path.join(config['model_path'], 'model_best.pth'))
                    log_string(LOG_FOUT, 'Model Saved!!!Utill now the best result is {}'.format(BEST_RES))
            
            
            if (iter_ + 1) % config['display'] == 0:
                time_end = time.time()    
                time_cost = time_end - time_start
                log_string(LOG_FOUT, 'iter: {} time: {}s lr: {} avg_loss: {} '.format(iter_ + 1, time_cost, config['lr'], loss_count / config['display']))
                loss_count = 0
                time_start = time.time()
            
            if (iter_ + 1) % config['test_iter'] == 0:
                RES = eval_metrics.eval_data_BOS(config)

            if (iter_ + 1) * config['batch_size'] % config['step_size'] == 0:
                if config['optim'] == 'SGD':
                    optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9, weight_decay=0.0005)
                    config['lr'] = config['lr'] * config['gamma']
                if config['optim'] == 'Adam':
                    optimizer = optim.Adam(net.parameters(), lr = config['lr'] * config['gamma'], weight_decay = 0.00005)  
                    config['lr'] = config['lr'] * config['gamma']
            

    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(config['model_path'],'CP{}.pth'.format(iter_ + 1)))
        log_string(LOG_FOUT, 'Saved interrupt pth: CP{}.pth'.format(iter_ + 1))
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def main():
    args = get_args()
    LOG_FOUT = open(os.path.join(args.model_path, 'log_train.txt'), 'a')

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    TRAIN_DST = FaceDataset(json_path = args.dset_json_path, hdf5_path = args.dset_hdf5_path, in_ch = args.in_ch, \
    batch_size = args.batch_size, input_size = args.input_size)
    
    NET = FaceNet(in_ch = args.in_ch, FRNet = args.FRNet, metric = args.loss, num_classes = args.num_classes)

    if args.restore:
        NET.load_state_dict(torch.load(args.load))
        log_string(LOG_FOUT, 'Model Restored from {}'.format(args.load))

    if args.gpu:
        NET.cuda()


    config = {}
    config['train_dset'] = TRAIN_DST
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
    config['test_iter'] = args.test_iter

    
    LOG_FOUT.write(str(config)+'\n')

    train_net(net = NET, gpu = args.gpu, config = config, LOG_FOUT = LOG_FOUT)


if __name__ == '__main__':
    main()