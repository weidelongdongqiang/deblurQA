#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:16:43 2020

@author: weizhe
"""
from CNNs import *
from utils import *
import os

# Load
preT='ckpt/step1/fusion_classify.pth'
ckpt='ckpt/step2'
load_dict=torch.load(preT)
D={k: v for k, v in load_dict.items() if 'classifier' not in k}
model=Siamese2()
model_dict=model.state_dict()
model_dict.update(D)
model.load_state_dict(model_dict)
model.to(torch.device('cuda'))
print('Successfully load pretrained model.')

# filter
for name, param in model.named_parameters():
    if 'fc5' in name or 'fc6' in name or 'gdn5' in name:
        param.requires_grad=False

# Train
loss_fn=Joint_loss2()
loss_rn=Regularization(model, 1.5e-5, p=1)
opt=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
Steps=100000
Lam=0.3
Del=0.1
trainGener=gen_multiscale('/media2/Data/deblurQA/Step2_train.txt',
                          folder='/media2/Data/deblurQA/deblur')
model.train()
for i in range(Steps):
    batch_X0,batch_Y0 = trainGener.__next__()
    batch_X1,batch_Y1 = trainGener.__next__()
    if batch_Y0 > batch_Y1:
        Y0=batch_Y0.cuda()
        Y1=batch_Y1.cuda()
        X0=batch_X0
        X1=batch_X1
    else:
        Y0=batch_Y1.cuda()
        Y1=batch_Y0.cuda()
        X0=batch_X1
        X1=batch_X0
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

# Save
torch.save(model.state_dict(), os.path.join(ckpt,'multiscale_4.pth'))
print('Model saved at %s' % os.path.join(ckpt,'multiscale_4.pth'))
