#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:53:12 2020

@author: weizhe
"""
import torch, os
import torch.nn as nn
from CNNs import *
from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader

vggPath='/home/weizhe/下载/vgg19-dcbb9e9d.pth'
batch=32
ckpt='ckpt/step1'
trans=transforms.Compose([transforms.RandomCrop(256),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                         ])
trainSet=deblurSet('/media2/Data/deblurQA/Step1_train_.txt', transform=trans)
train_loader=DataLoader(trainSet, batch, shuffle=True)
valSet=deblurSet('/media2/Data/deblurQA/Step1_val_.txt', transform=trans)
val_loader=DataLoader(valSet, 100, shuffle=True)

# Load
model=Integrate()
model_dict=model.state_dict()
vgg_dict=torch.load(vggPath)
Map={'myVgg19.conv1':'features.0', 'myVgg19.conv2':'features.2', 'myVgg19.conv3': 'features.5',
     'myVgg19.conv4': 'features.7', 'myVgg19.conv5': 'features.10', 'myVgg19.conv6': 'features.12',
     'myVgg19.conv7': 'features.14', 'myVgg19.conv8': 'features.16'}
D={}
for k in model_dict.keys():
    if 'myVgg19' in k:
        newkey=Map[k[:13]]+k[13:]
        D[k]=vgg_dict[newkey]
model_dict.update(D)
model.load_state_dict(model_dict)
print('Successfully load VGG!')
model.to(torch.device('cuda'))

# Frozen
for param in model.myVgg19.parameters():
    param.requires_grad=False

# Train
loss_cn=nn.BCEWithLogitsLoss()
loss_rn=Regularization(model, 1.5e-7, p=1)
opt=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1.0e-4)
epochs=50
step=0
for i in range(epochs):
    for batch_X,batch_Y in iter(train_loader):
        X=batch_X.cuda()
        Y=batch_Y.cuda()
        model.train()
        # forward
        Y_=model(X)
        loss_c=loss_cn(Y_, Y)
        loss_r=loss_rn()
        loss=loss_c+loss_r
        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 ==0:
            batch_X,batch_Y = iter(val_loader).__next__()
            X=batch_X.cuda()
            Y=batch_Y.cuda()
            model.eval()
            Y_=model(X)
            logits=torch.sigmoid(Y_.detach())>0.5
            correct=torch.eq(logits, Y)
            rate=correct.float().mean()
            print('Epoch :%d, Step: %d, Loss: %.4f, Class loss: %.4f, Correct: %.4f' 
                  % (i, step, loss.data, loss_c.data, rate.data))
        step+=1
stat=model.state_dict()
torch.save(stat, os.path.join(ckpt,'integ_classify.pth'))
print('Model saved at %s' % os.path.join(ckpt,'integ_classify.pth'))
