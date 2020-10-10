#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:02:20 2020

@author: weizhe
"""
import torch, os, cv2
from CNNs import *
import numpy as np
import scipy.io as sio

# Load
ckpt='ckpt/step2'
# model=Model1()
model=Integrate()
model.load_state_dict(torch.load(os.path.join(ckpt, 'l1.pth')))
model.to(torch.device('cuda'))
print('Successfully load model.')

Folder='/media2/Data/deblurQA/deblur'
with open('/media2/Data/deblurQA/Step2_test.txt') as fr:
    Lines=fr.readlines()
GT=[]
Predicted=[]
model.eval()
with torch.no_grad():
    for line in Lines:
        l=line.strip('\n')
        fName,score_s=l.split(' ')
        img=cv2.imread(os.path.join(Folder,fName))/127.5-1
        img=img[:,:,::-1].copy()
        score=float(score_s)
        Pred=[]
        H,W,_=img.shape
        for i in range(10):
            offy=np.random.randint(0,H-256)
            offx=np.random.randint(0,W-256)
            crop=img[offy:offy+256, offx:offx+256]
            subimg=np.expand_dims(crop,0)
            subimg=np.transpose(subimg,[0,3,1,2])
            subimg_T=torch.Tensor(subimg).cuda()
            pred=model(subimg_T)
            Pred.append(pred.detach().cpu().numpy())
        GT.append(score)
        Predicted.append(np.mean(Pred))
        print(l)
    
# Save
sFold='output/model3'
if not os.path.exists(sFold):
    os.makedirs(sFold)
sio.savemat(os.path.join(sFold,'l1.mat'), {'GT': GT, 'Predicted': Predicted})
print('Saved')
