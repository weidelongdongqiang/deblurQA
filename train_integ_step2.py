#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:16:43 2020

@author: weizhe
"""
from CNNs import *
from utils import *
import os
from torchvision import transforms
from torch.utils.data import DataLoader

# Load
preT='ckpt/step1/integ_classify.pth'
ckpt='ckpt/step2'
load_dict=torch.load(preT)
D={k: v for k, v in load_dict.items() if 'classifier' not in k}
model=Siamese3()
model_dict=model.state_dict()
model_dict.update(D)
model.load_state_dict(model_dict)
model.to(torch.device('cuda'))
print('Successfully load pretrained model.')

# Frozen
for param in model.myVgg19.parameters():
    param.requires_grad=False

# Train
# loss_fn=Joint_loss2()
# loss_fn=Joint_loss3()
# loss_fn=Joint_loss1()
# loss_fn=nn.SmoothL1Loss()
loss_fn=nn.L1Loss()
loss_rn=Regularization(model, 1.0e-6, p=1)
opt=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
Steps=100000
Lam=0.6
Del=0.06
trans=transforms.Compose([transforms.RandomCrop(256),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                         ])
trainSet=deblurSet('/media2/Data/deblurQA/Step2_train.txt', transform=trans,
                   folder='/media2/Data/deblurQA/deblur')
train_loader=DataLoader(trainSet, 2, shuffle=True)

model.train()
for i in range(Steps):
    X,Y=next(iter(train_loader))
    if Y[0]>=Y[1]:
        Y0=Y[0].float().unsqueeze(0).cuda()
        Y1=Y[1].float().unsqueeze(0).cuda()
        X0=X[0].unsqueeze(0).cuda()
        X1=X[1].unsqueeze(0).cuda()
    else:
        Y0=Y[1].float().unsqueeze(0).cuda()
        Y1=Y[0].float().unsqueeze(0).cuda()
        X0=X[1].unsqueeze(0).cuda()
        X1=X[0].unsqueeze(0).cuda()
    # forward
    Y0_,Y1_=model(X0,X1)
    # loss1=loss_fn(Y0,Y1,Y0_,Y1_,lamda=Lam,delta=Del)
    # loss1=loss_fn(Y0,Y1,Y0_,Y1_,lamda=Lam)
    loss1=loss_fn(Y0,Y0_)+loss_fn(Y1,Y1_)
    loss2=loss_rn()
    loss=loss1+loss2
    # Backward
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i%50 ==0:
        print('[Train]: step: %d, loss: %.4f, joint loss: %.4f' % (i,loss.data,loss1.data))

# Save
torch.save(model.state_dict(), os.path.join(ckpt,'l1.pth'))
print('Model saved at %s' % os.path.join(ckpt,'l1.pth'))
