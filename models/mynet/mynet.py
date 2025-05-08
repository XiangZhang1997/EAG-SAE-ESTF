# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from module import DoubleConv, Up, OutConv, CoronaryIdentificationModule, MultiScaleConvattModule, LongDistanceDependencyModule_onlytrans


class EAG_SAE_ESTF(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, depths=[1, 1, 1, 1, 1],c_num=[32,64,128,256,512]):
        super(EAG_SAE_ESTF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.f_num = c_num
        self.sigmoid = nn.Sigmoid()

        # encoder
        resnet = models.resnet34(pretrained=True)
        weights = resnet.state_dict() 
        weights['conv1.weight'] = weights['conv1.weight'].sum(1, keepdim=True)  
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        resnet.load_state_dict(weights)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.conv0 = nn.Conv2d(64,self.f_num[0],1)

        # cim
        self.cim0 = CoronaryIdentificationModule(self.f_num[0],self.f_num[0]) # 32
        self.cim1 = CoronaryIdentificationModule(self.f_num[1],self.f_num[1]) # 64
        self.cim2 = CoronaryIdentificationModule(self.f_num[2],self.f_num[2]) # 128
        self.cim3 = CoronaryIdentificationModule(self.f_num[3],self.f_num[3]) # 256
        self.cim4 = CoronaryIdentificationModule(self.f_num[4],self.f_num[4]) # 512

        # ms
        self.ms = MultiScaleConvattModule(self.f_num[4],self.f_num[4])
        # lrdm
        self.lddm = LongDistanceDependencyModule_onlytrans(self.f_num[4],H=16,W=16)
        # self.lddm = LongDistanceDependencyModule_onlytrans(self.f_num[4],H=10,W=10) # for DCA1

        # decoder
        self.up1 = Up(self.f_num[4], self.f_num[3])
        self.dblock1 = DoubleConv(self.f_num[4], self.f_num[3])
        self.up2 = Up(self.f_num[3], self.f_num[2])
        self.dblock2 = DoubleConv(self.f_num[3], self.f_num[2])
        self.up3 = Up(self.f_num[2], self.f_num[1])
        self.dblock3 = DoubleConv(self.f_num[2], self.f_num[1])
        self.up4 = Up(self.f_num[1], self.f_num[0])
        self.dblock4 = DoubleConv(self.f_num[1], self.f_num[0])

        self.side_up4 = OutConv(self.f_num[0], n_classes)


    def forward(self, x):
        output_dict = {} 

        rx = x
        x1 = self.firstconv(rx) 
        x1 = self.firstbn(x1) 
        x1 = self.firstrelu(x1) 
        px1 = self.firstmaxpool(x1) 
        x2 = self.encoder1(px1) 
        x3 = self.encoder2(x2) 
        x4 = self.encoder3(x3) 
        x5 = self.encoder4(x4) 
        x1_ = self.conv0(x1) 

        cim1 = self.cim0(x1_)
        cim2 = self.cim1(x2)
        cim3 = self.cim2(x3)
        cim4 = self.cim3(x4)
        cim5 = self.cim4(x5)

        ms = self.ms(cim5)
        lddm = self.lddm(ms)

        u1 = self.up1(lddm)
        u1 = torch.cat([cim4, u1], dim=1)
        d1 = self.dblock1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([cim3, u2], dim=1)
        d2 = self.dblock2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([cim2, u3], dim=1)
        d3 = self.dblock3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([cim1, u4], dim=1)
        d4 = self.dblock4(u4)

        logits = self.side_up4(d4)
        out = self.sigmoid(logits)
        output_dict['output'] = out

        return output_dict

from calflops import calculate_flops

if __name__ == '__main__':
    H = W = 256
    batch_size = 2
    C = 1
    x = torch.randn((batch_size, C, H, W)).cuda()
    model = EAG_SAE_ESTF().cuda()
    flops, macs, params = calculate_flops(model=model, input_shape=(batch_size, C, H, W))
    print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))