import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *

class FaceNet(nn.Module):

    def __init__(self,in_ch = 3, FRNet = 'resnet18', metric = None,num_classes = None, easy_margin = False, large_conv = False):
        super(FaceNet, self).__init__()

        self.large_conv = large_conv
        if self.large_conv:

            self.conv1 = conv_activation(in_ch = in_ch, out_ch = 64, kernel_size=7, stride=1 ,padding= 3)
            self.conv2 = conv_activation(in_ch = 64, out_ch = 64, kernel_size=7, stride=1 ,padding= 3)

            if FRNet == 'resnet50':
                self.FRNet = resnet50(in_ch = 64)
            elif FRNet == 'resnet34':
                self.FRNet = resnet34(in_ch = 64)
            elif FRNet == 'resnet18':
                self.FRNet = resnet18(pretrained = False,in_ch = 64)
                self.FRNet.train(mode=True)
            elif FRNet == 'resnet101':
                self.FRNet = resnet101(in_ch = 64)
        
        else:

            if FRNet == 'resnet50':
                self.FRNet = resnet50(in_ch = in_ch)
            elif FRNet == 'resnet34':
                self.FRNet = resnet34(in_ch = in_ch)
            elif FRNet == 'resnet18':
                self.FRNet = resnet18(in_ch = in_ch)
                self.FRNet.train(mode=True)
            elif FRNet == 'resnet101':
                self.FRNet = resnet101(in_ch = in_ch)


        if metric == 'AddMarginLoss':
            self.metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
        elif metric == 'ArcMarginLoss':
            self.metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin = easy_margin)
        elif metric == 'SphereLoss':
            self.metric_fc = SphereProduct(512, num_classes, m=4)
        elif metric == 'Linear' :
            self.metric_fc = nn.Linear(512, num_classes)

    def forward(self, data, label = None,mode = 'train'):
        if mode == 'train':
            if self.large_conv:
                conv1 = self.conv1(data)
                conv2 = self.conv2(conv1)
                feature = self.FRNet(conv2)
                output = self.metric_fc(feature,label)
                return output
            else:
                feature = self.FRNet(data)
                output = self.metric_fc(feature,label)
                return output

        elif mode == 'test':
            self.FRNet.eval()  
            if self.large_conv:   
                conv1 = self.conv1(data)
                conv2 = self.conv2(conv1)
                feature = self.FRNet(conv2)
                return feature
            else:
                feature = self.FRNet(data)
                return feature