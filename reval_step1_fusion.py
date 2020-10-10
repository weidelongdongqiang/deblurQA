#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:42:35 2020

@author: weizhe
"""
# Validate fusion step1 training
from CNNs import *
from utils import *

modelPath='ckpt/step1/fusion_classify_.pth'
model=Fusion()
model.load_state_dict(torch.load(modelPath))
model.to(torch.device('cuda'))
print('Successfully load model.')

model.eval()
valGener=gen_multiscale('/media2/Data/deblurQA/Step1_val_.txt')
Judge=list()
for i in range(280):
    batch_X,batch_Y = valGener.__next__()
    X=batch_X
    Y=batch_Y.cuda()
    Y_=model(X)
    logits=torch.sigmoid(Y_.detach())>0.5
    correct=torch.eq(logits, Y)
    if correct.cpu().numpy()[0]>0:
        Judge.append(1)
    else:
        Judge.append(0)
    print(i)
print('Mean correct rate is ', np.mean(Judge))
    