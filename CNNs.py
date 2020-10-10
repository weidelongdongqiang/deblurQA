#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:41:58 2020

@author: weizhe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_gdn import GDN
import math

# From MEON for class
# Input: 256
# with gdn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,8, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(8,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(16,device)
        self.conv3=nn.Conv2d(16, 32, (5,5), stride=(2,2), padding=(2,2))
        self.gdn3=GDN(32,device)
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(64,device)
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.gdn5=GDN(128,device)
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.drop=nn.Dropout(0.5)
        
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
        # 5
        hidden = self.fc5(hidden)
        hidden = self.drop(hidden)
        hidden = self.gdn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# with LN
class Model1_1(nn.Module):
    def __init__(self):
        super(Model1_1, self).__init__()
        self.conv1=nn.Conv2d(3,8, (5,5), stride=(2,2), padding=(2,2))
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (5,5), stride=(2,2), padding=(2,2))
        self.conv3=nn.Conv2d(16, 32, (5,5), stride=(2,2), padding=(2,2))
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.ln1=nn.LayerNorm([8,128,128])
        self.ln2=nn.LayerNorm([16,32,32])
        self.ln3=nn.LayerNorm([32,8,8])
        self.ln4=nn.LayerNorm([64,2,2])
        self.ln5=nn.LayerNorm([128,1,1])
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.ln1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.ln2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.ln3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.ln4(hidden)
        hidden = self.pool(hidden)
        # 5
        hidden = self.fc5(hidden)
        hidden = self.ln5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# with GN
class Model1_2(nn.Module):
    def __init__(self):
        super(Model1_2, self).__init__()
        self.conv1=nn.Conv2d(3,8, (5,5), stride=(2,2), padding=(2,2))
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (5,5), stride=(2,2), padding=(2,2))
        self.conv3=nn.Conv2d(16, 32, (5,5), stride=(2,2), padding=(2,2))
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.gn1=nn.GroupNorm(2,8)
        self.gn2=nn.GroupNorm(4,16)
        self.gn3=nn.GroupNorm(8,32)
        self.gn4=nn.GroupNorm(16,64)
        self.gn5=nn.GroupNorm(32,128)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gn4(hidden)
        hidden = self.pool(hidden)
        # 5
        hidden = self.fc5(hidden)
        hidden = self.gn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# From MEON for class
# Input: 32
# with gdn
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,8, (3,3), stride=(1,1), padding=(1,1))
        self.gdn1=GDN(8,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (3,3), stride=(1,1), padding=(1,1))
        self.gdn2=GDN(16,device)
        self.conv3=nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=(1,1))
        self.gdn3=GDN(32,device)
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(64,device)
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.gdn5=GDN(128,device)
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.drop=nn.Dropout(0.5)
        
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
        # 5
        hidden = self.fc5(hidden)
        hidden = self.drop(hidden)
        hidden = self.gdn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# From MEON for class
# Input: 128
# with gdn
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,8, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(8,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(16,device)
        self.conv3=nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=(1,1))
        self.gdn3=GDN(32,device)
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(64,device)
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.gdn5=GDN(128,device)
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.drop=nn.Dropout(0.5)
        
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
        # 5
        hidden = self.fc5(hidden)
        hidden = self.drop(hidden)
        hidden = self.gdn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# From MEON for class
# Input: 64
# with gdn
class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,8, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(8,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8, 16, (3,3), stride=(1,1), padding=(1,1))
        self.gdn2=GDN(16,device)
        self.conv3=nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=(1,1))
        self.gdn3=GDN(32,device)
        self.conv4=nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(64,device)
        self.fc5=nn.Conv2d(64, 128, (1,1), stride=(1,1), padding=(0,0))
        self.gdn5=GDN(128,device)
        self.fc6=nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0))
        self.drop=nn.Dropout(0.5)
        
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
        # 5
        hidden = self.fc5(hidden)
        hidden = self.drop(hidden)
        hidden = self.gdn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

# Multi-scale CNN fusion
# with model1~4 pretrained
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        device = torch.device('cuda')
        self.model256=Model1()
        self.model32=Model2()
        self.model128=Model3()
        self.model64=Model4()
        self.classifier=nn.Sequential(
            nn.Conv2d(5440,128,(1,1),stride=(1,1),padding=0),
            nn.Conv2d(128, 1, (1,1), stride=(1,1), padding=(0,0)))
    
    def forward256(self, X):
        # 1
        hidden = self.model256.conv1(X)
        hidden = self.model256.gdn1(hidden)
        hidden = self.model256.pool(hidden)
        # 2
        hidden = self.model256.conv2(hidden)
        hidden = self.model256.gdn2(hidden)
        hidden = self.model256.pool(hidden)
        # 3
        hidden = self.model256.conv3(hidden)
        hidden = self.model256.gdn3(hidden)
        hidden = self.model256.pool(hidden)
        # 4
        hidden = self.model256.conv4(hidden)
        hidden = self.model256.gdn4(hidden)
        hidden = self.model256.pool(hidden)
        return hidden
        
    def forward128(self, X):
        # 1
        hidden = self.model128.conv1(X)
        hidden = self.model128.gdn1(hidden)
        hidden = self.model128.pool(hidden)
        # 2
        hidden = self.model128.conv2(hidden)
        hidden = self.model128.gdn2(hidden)
        hidden = self.model128.pool(hidden)
        # 3
        hidden = self.model128.conv3(hidden)
        hidden = self.model128.gdn3(hidden)
        hidden = self.model128.pool(hidden)
        # 4
        hidden = self.model128.conv4(hidden)
        hidden = self.model128.gdn4(hidden)
        hidden = self.model128.pool(hidden)
        return hidden

    def forward64(self, X):
        # 1
        hidden = self.model64.conv1(X)
        hidden = self.model64.gdn1(hidden)
        hidden = self.model64.pool(hidden)
        # 2
        hidden = self.model64.conv2(hidden)
        hidden = self.model64.gdn2(hidden)
        hidden = self.model64.pool(hidden)
        # 3
        hidden = self.model64.conv3(hidden)
        hidden = self.model64.gdn3(hidden)
        hidden = self.model64.pool(hidden)
        # 4
        hidden = self.model64.conv4(hidden)
        hidden = self.model64.gdn4(hidden)
        hidden = self.model64.pool(hidden)
        return hidden

    def forward32(self, X):
        # 1
        hidden = self.model32.conv1(X)
        hidden = self.model32.gdn1(hidden)
        hidden = self.model32.pool(hidden)
        # 2
        hidden = self.model32.conv2(hidden)
        hidden = self.model32.gdn2(hidden)
        hidden = self.model32.pool(hidden)
        # 3
        hidden = self.model32.conv3(hidden)
        hidden = self.model32.gdn3(hidden)
        hidden = self.model32.pool(hidden)
        # 4
        hidden = self.model32.conv4(hidden)
        hidden = self.model32.gdn4(hidden)
        hidden = self.model32.pool(hidden)
        return hidden

    def forward(self, X):
        # input: X-----85-long list containing [1,3,size,size] images
        # size: 256(1),128(4),64(16),32(64)
        feature=self.forward256(X[0].cuda())
        for i in range(1,5):
            temp=self.forward128(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        for i in range(5, 21):
            temp=self.forward64(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        for i in range(21, 85):
            temp=self.forward32(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        hidden=self.classifier(feature)
        return hidden.view(hidden.size(0))

# Model3 usage
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

# Model1 siamese
class Siamese1(Model1):
    def __init__(self):
        super(Siamese1, self).__init__()
    
    def forward_once(self, X):
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
        # 5
        hidden = self.fc5(hidden)
        hidden = self.gdn5(hidden)
        # 6
        hidden = self.fc6(hidden)
        return hidden.view(hidden.size(0))

    def forward(self, X1, X2):
        Y1=self.forward_once(X1)
        Y2=self.forward_once(X2)
        return Y1, Y2

# Fusion siamese
class Siamese2(Fusion):
    def __init__(self):
        super(Siamese2, self).__init__()
    
    def forward_once(self, X):
        # input: X-----85-long list containing [1,3,size,size] images
        # size: 256(1),128(4),64(16),32(64)
        feature=self.forward256(X[0].cuda())
        for i in range(1,5):
            temp=self.forward128(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        for i in range(5, 21):
            temp=self.forward64(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        for i in range(21, 85):
            temp=self.forward32(X[i].cuda())
            feature=torch.cat((feature,temp), 1)
        hidden=self.classifier(feature)
        return hidden.view(hidden.size(0))
    
    def forward(self, X1, X2):
        Y1=self.forward_once(X1)
        Y2=self.forward_once(X2)
        return Y1,Y2

# Integrated siamese
class Siamese3(Integrate):
    def __init__(self):
        super(Siamese3, self).__init__()
    
    def forward_once(self, X):
        B128,B64,B32=self.myVgg19(X)
        feature256=self.branch256(X)
        feature128=self.branch128(B128)
        feature64=self.branch64(B64)
        feature32=self.branch32(B32)
        Integ=torch.cat((feature256,feature128,feature64,feature32), 1)
        hidden=self.classifier(Integ)
        return hidden.view(hidden.size(0))
    
    def forward(self, X1, X2):
        Y1=self.forward_once(X1)
        Y2=self.forward_once(X2)
        return Y1,Y2

# Regularization loss
class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight()
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self):
        # self.weight_list=self.get_weight(self.model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
    
    def get_weight(self):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
    
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

# step2: Joint loss1: Y---real label, Y_---predicted
# Y1>Y2
class Joint_loss1(nn.Module):
    def __init__(self):
        super(Joint_loss1, self).__init__()
        return
    
    def forward(self, Y1, Y2, Y1_, Y2_, lamda, delta):
        loss1=torch.abs(Y1-Y1_)+torch.abs(Y2-Y2_)
        loss2=torch.clamp(Y2_-Y1_+delta, min=0)
        # loss2=torch.max(torch.Tensor([0]).cuda(), Y2_-Y1_+delta)
        return loss1+lamda*loss2

class Joint_loss2(nn.Module):
    def __init__(self):
        super(Joint_loss2, self).__init__()
        return
    
    def forward(self, Y1, Y2, Y1_, Y2_, lamda, delta):
        loss1=F.smooth_l1_loss(Y1_, Y1)+F.smooth_l1_loss(Y2_, Y2)
        loss2=torch.clamp(Y2_-Y1_+delta, min=0)
        return loss1+lamda*loss2

class Joint_loss3(nn.Module):
    def __init__(self):
        super(Joint_loss3, self).__init__()
        return
    
    def forward(self, Y1, Y2, Y1_, Y2_, lamda, Tc=30.):
        loss1=F.smooth_l1_loss(Y1_, Y1)+F.smooth_l1_loss(Y2_, Y2)
        if Y1-Y2<=Tc:
            U=0.5*(1+torch.cos(math.pi*(Y1-Y2)/Tc))
        else:
            U=0
        L=-F.logsigmoid(Y1_-Y2_)
        loss2=(1-U)*L
        return loss1+lamda*loss2
