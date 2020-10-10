#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:24:48 2020

@author: weizhe
"""
from CNNs import Fusion
import numpy as np
import scipy.io as sio
import torch, os, cv2

ckpt='ckpt/step2'
model=Fusion()
model.load_state_dict(torch.load(os.path.join(ckpt, 'multiscale_0.pth')))
model.to(torch.device('cuda'))
print('Successfully load model.')

Folder='/media2/Data/deblurQA/deblur'
with open('/media2/Data/deblurQA/Step2_test.txt') as fr:
    Lines=fr.readlines()
GT=[]
Predicted=[]
model.eval()
size=[256,128,64,32]
for line in Lines:
    l=line.strip('\n')
    fName,score_s=l.split(' ')
    score=float(score_s)
    img=cv2.imread(os.path.join(Folder,fName))/127.5-1
    img=img[:,:,::-1].copy()
    H,W,_ = img.shape
    Pred=[]
    for k in range(10):
        roll=[]
        for i in range(len(size)):
            cut_y=H//(2**i)
            cut_x=W//(2**i)
            for y_int in range(2**i):
                y_low=y_int*cut_y
                y_high=(y_int+1)*cut_y-size[i]
                start_y=np.random.randint(y_low, y_high)
                for x_int in range(2**i):
                    x_low=x_int*cut_x
                    x_high=(x_int+1)*cut_x-size[i]
                    start_x=np.random.randint(x_low, x_high)
                    crop=img[start_y:start_y+size[i],start_x:start_x+size[i],:].copy()
                    crop=np.expand_dims(crop, 0)
                    crop=np.transpose(crop, [0,3,1,2])
                    roll.append(torch.Tensor(crop))
        pred=model(roll)
        Pred.append(pred.detach().cpu().numpy())
    GT.append(score)
    Predicted.append(np.mean(Pred))
    print(l)

# Save
sFold='output/model2'
if not os.path.exists(sFold):
    os.makedirs(sFold)
sio.savemat(os.path.join(sFold,'joint_loss2.mat'), {'GT': GT, 'Predicted': Predicted})
print('Saved')
