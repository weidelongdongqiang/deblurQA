#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:01:07 2020

@author: weizhe
"""
import torch, os
import torch.nn as nn
from CNNs import *
from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader

preTrained='ckpt/step1/256.pth'
ckpt='ckpt/step2'
if not os.path.exists(ckpt):
    os.makedirs(ckpt)
trans=transforms.Compose([transforms.RandomCrop(256),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                         ])
trainSet=deblurSet('/media2/Data/deblurQA/Step2_train.txt', transform=trans,
                   folder='/media2/Data/deblurQA/deblur')
train_loader=DataLoader(trainSet, 2, shuffle=True)
valSet=deblurSet('/media2/Data/deblurQA/Step2_val.txt', transform=trans,
                 folder='/media2/Data/deblurQA/deblur')
val_loader=DataLoader(valSet, 2, shuffle=True)

# Load pretrained 1~4
stat=torch.load(preTrained)
preT_dict=stat['model']
load_dict={k: v for k, v in preT_dict.items() if '5' not in k and '6' not in k}
model=Siamese1()
model_dict=model.state_dict()
model_dict.update(load_dict)
model.load_state_dict(model_dict)
model.to(torch.device('cuda'))
print('Successfully load pretrained model.')

# Train
loss_fn=Joint_loss1()
loss_rn=Regularization(model, 5.e-4, p=1)
opt=torch.optim.Adam(model.parameters(), lr=1e-4)
Steps=200000
Lam=0.3
Del=0.06
for i in range(Steps):
    X,Y=next(iter(train_loader))
    if Y[0]>=Y[1]:
        Y0=Y[0].cuda()
        Y1=Y[1].cuda()
        X0=X[0].unsqueeze(0).cuda()
        X1=X[1].unsqueeze(0).cuda()
    else:
        Y0=Y[1].cuda()
        Y1=Y[0].cuda()
        X0=X[1].unsqueeze(0).cuda()
        X1=X[0].unsqueeze(0).cuda()
    model.train()
    # forward
    Y0_,Y1_=model(X0,X1)
    loss1=loss_fn(Y0,Y1,Y0_,Y1_,lamda=Lam,delta=Del)
    loss2=loss_rn()
    loss=loss1+loss2
    # Backward
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i%50 ==0:
        print('[Train]: step: %d, loss: %.4f, joint loss: %.4f' % (i,loss.data,loss1.data))
    if i%500 == 0:
        # validate
        X,Y=next(iter(val_loader))
        if Y[0]>=Y[1]:
            Y0=Y[0].cuda()
            Y1=Y[1].cuda()
            X0=X[0].unsqueeze(0).cuda()
            X1=X[1].unsqueeze(0).cuda()
        else:
            Y0=Y[1].cuda()
            Y1=Y[0].cuda()
            X0=X[1].unsqueeze(0).cuda()
            X1=X[0].unsqueeze(0).cuda()
        model.eval()
        Y0_,Y1_=model(X0,X1)
        loss1=loss_fn(Y0,Y1,Y0_,Y1_,lamda=Lam,delta=Del)
        loss2=loss_rn()
        loss=loss1+loss2
        print('[Validate]: step: %d, loss: %.4f, joint loss: %.4f' % (i,loss.data,loss1.data))

# Save
torch.save(model.state_dict(), os.path.join(ckpt,'MEON_256_10.pth'))
print('Model saved at %s' % ckpt)
