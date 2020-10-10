#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:21:23 2020

@author: weizhe
"""
import torch, os
from CNNs import *
from PIL import Image
import numpy as np

ckpt='ckpt/step1'
Size=256
with open('/media2/Data/deblurQA/Step1_val_.txt') as f:
    Lines=f.readlines()

def transform(img):
    # img: 0~1
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        raise ValueError('Image should have 3 channels.')
    offx=np.random.randint(0,W-Size)
    offy=np.random.randint(0,H-Size)
    crop=img[offy:offy+Size, offx:offx+Size, :]
    return crop*2-1

# Load
stat=torch.load(os.path.join(ckpt,'integ_classify.pth'))
model=Integrate()
model.load_state_dict(stat)
model.to(torch.device('cuda'))
model.eval()

wrongLines=[]
for l in Lines:
    Names,Labs = l.strip('\n').split(' ')
    im=np.array(Image.open(Names).convert('RGB'))
    im=transform(im/255.)
    img=np.expand_dims(im,0)
    img=np.transpose(img,[0,3,1,2])
    lab=np.array([int(Labs)])
    imT=torch.Tensor(img).cuda()
    labT=torch.Tensor(lab).cuda()
    pred=model(imT)
    logits=torch.sigmoid(pred)>0.5
    correct=torch.eq(logits, labT)
    if ~correct.data[0]:
        wrongLines.append(l)
    print(l.strip('\n'))
# if not os.path.exists('output'):
#     os.makedirs('output')
# with open('output/clas_wrong_256.txt','w') as fw:
#     fw.writelines(wrongLines)
print('Accuracy: %.3f' % (1-len(wrongLines)/len(Lines)))
