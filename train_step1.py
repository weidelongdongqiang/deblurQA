#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:47:56 2020

@author: weizhe
"""
import torch, os
import torch.nn as nn
from CNNs import *
from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader

batch=32
ckpt='ckpt/step1'
if not os.path.exists(ckpt):
    os.makedirs(ckpt)
trans=transforms.Compose([transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                         ])
trainSet=deblurSet('/media2/Data/deblurQA/Step1_train_.txt', transform=trans)
train_loader=DataLoader(trainSet, batch, shuffle=True)
valSet=deblurSet('/media2/Data/deblurQA/Step1_val_.txt', transform=trans)
val_loader=DataLoader(valSet, 100, shuffle=True)

model=Model2().cuda()
loss_cn=nn.BCEWithLogitsLoss()
loss_rn=Regularization(model, 5.e-6, p=1)
opt=torch.optim.Adam(model.parameters(), lr=1.0e-4)
# opt=torch.optim.RMSprop(model.parameters())
# opt=torch.optim.Adagrad(model.parameters())
epochs=100
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
stat={'model':model.state_dict(), 'opt':opt.state_dict(), 'epoch':epochs}
torch.save(stat, os.path.join(ckpt,'32_1.pth'))
print('Model saved at %s' % ckpt)
