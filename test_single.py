#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:02:20 2020

@author: weizhe
"""
import torch, os, cv2
from model import *
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Your image to be assess")
parser.add_argument('--image', required=True, type=str)
args = parser.parse_args()

# Load
ckpt='Model.pth'
# model=Model1()
model=Integrate()
model.load_state_dict(torch.load(ckpt))
model.to(torch.device('cuda'))
print('Successfully load model.')

model.eval()
Pred=[]
with torch.no_grad():
    img=cv2.imread(args.image)/127.5-1
    img=img[:,:,::-1].copy()
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
Score=np.mean(Pred)

print('%s scores: %.4f' % (args.image, Score))
