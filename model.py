#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:41:58 2020

@author: weizhe
"""
import torch
import torch.nn as nn
from pytorch_gdn import GDN

# VGG19
class myVgg19(nn.Module):
    def __init__(self):
      super(myVgg19, self).__init__()
      self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1)
      self.actfunc=nn.ReLU()
      self.conv2=nn.Conv2d(64,64,3,stride=1,padding=1)
      self.pool=nn.MaxPool2d(2,stride=2)
      self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
      self.conv4=nn.Conv2d(128,128,3,stride=1,padding=1)
      self.conv5=nn.Conv2d(128,256,3,stride=1,padding=1)
      self.conv6=nn.Conv2d(256,256,3,stride=1,padding=1)
      self.conv7=nn.Conv2d(256,256,3,stride=1,padding=1)
      self.conv8=nn.Conv2d(256,256,3,stride=1,padding=1)

    def forward(self, X):
        # 1
        hidden=self.conv1(X)
        hidden=self.actfunc(hidden)
        hidden=self.conv2(hidden)
        hidden=self.actfunc(hidden)
        out1=self.pool(hidden)
        # 2
        hidden=self.conv3(out1)
        hidden=self.actfunc(hidden)
        hidden=self.conv4(hidden)
        hidden=self.actfunc(hidden)
        out2=self.pool(hidden)
        # 3
        hidden=self.actfunc(self.conv5(out2))
        hidden=self.actfunc(self.conv6(hidden))
        hidden=self.actfunc(self.conv7(hidden))
        hidden=self.actfunc(self.conv8(hidden))
        out3=self.pool(hidden)
        return out1, out2, out3

# MEON256 strenthen
# input: 256,256,3
class branch256(nn.Module):
    def __init__(self):
        super(branch256, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,32, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(32,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32, 64, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(64,device)
        self.conv3=nn.Conv2d(64, 128, (5,5), stride=(2,2), padding=(2,2))
        self.gdn3=GDN(128,device)
        self.conv4=nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gdn4(hidden)
        hidden = self.pool(hidden)
        return hidden

# MEON128 strenthen
# input: 128,128,64
class branch128(nn.Module):
    def __init__(self):
        super(branch128, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(64,128, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(128,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))
        self.gdn3=GDN(256,device)
        self.conv4=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gdn4(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 64,64,128
class branch64(nn.Module):
    def __init__(self):
        super(branch64, self).__init__()
        device = torch.device('cuda')
        self.pool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(256,device)
        self.conv2=nn.Conv2d(256, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 32,32,256
class branch32(nn.Module):
    def __init__(self):
        super(branch32, self).__init__()
        device = torch.device('cuda')
        self.pool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(256, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(256,device)
        self.conv2=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# integrated CNN
# with vgg19 pretrained
class Integrate(nn.Module):
    def __init__(self):
        super(Integrate, self).__init__()
        self.myVgg19=myVgg19()
        self.branch256=branch256()
        self.branch128=branch128()
        self.branch64=branch64()
        self.branch32=branch32()
        self.classifier=nn.Sequential(
            nn.Conv2d(1024,256,(1,1),stride=(1,1),padding=0),
            nn.Dropout(0.5),
            nn.Conv2d(256, 1, (1,1), stride=(1,1), padding=(0,0))
            )
    
    def forward(self, X):
        B128,B64,B32=self.myVgg19(X)
        feature256=self.branch256(X)
        feature128=self.branch128(B128)
        feature64=self.branch64(B64)
        feature32=self.branch32(B32)
        Integ=torch.cat((feature256,feature128,feature64,feature32), 1)
        hidden=self.classifier(Integ)
        return hidden.view(hidden.size(0))

