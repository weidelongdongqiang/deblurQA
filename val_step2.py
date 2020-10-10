#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:47:46 2020

@author: weizhe
"""
import torch, os
from CNNs import *
from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

ckpt='ckpt/step2'
trans=transforms.Compose([transforms.RandomCrop(256),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                         ])
valSet=deblurSet('/media2/Data/deblurQA/Step2_val.txt', transform=trans,
                 folder='/media2/Data/deblurQA/deblur')
val_loader=DataLoader(valSet, 1, shuffle=True)

# Load
model=Integrate()
model.load_state_dict(torch.load(os.path.join(ckpt, 'integrate_loss1_5.pth')))
model.to(torch.device('cuda'))
print('Successfully load model.')
model.eval()

# Validate
error=list()
with torch.no_grad():
    for i in range(len(val_loader)):
        X,Y=next(iter(val_loader))
        X=X.cuda()
        Y=Y.cuda()
        Y_=model(X)
        Er=torch.abs(Y-Y_)
        error.append(Er.cpu().detach().numpy())
        print('{} is finished'.format(i))

print('Mean square error: %.4f' % np.mean(np.array(error)**2))
