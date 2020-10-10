#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:31:12 2020

@author: weizhe
"""
from CNNs import *
from utils import *

ckpt='ckpt/step2'
valGener=gen_multiscale('/media2/Data/deblurQA/Step2_val.txt',
                          folder='/media2/Data/deblurQA/deblur')
model=Fusion()
model.load_state_dict(torch.load(os.path.join(ckpt, 'multiscale_4.pth')))
model.to(torch.device('cuda'))
print('Successfully load model.')

# Validate
model.eval()
error=list()
for i in range(300):
    batch_X,batch_Y = valGener.__next__()
    Y=batch_Y.cuda()
    Y_=model(batch_X)
    Er=torch.abs(Y-Y_)
    error.append(Er.cpu().detach().numpy())
    print('{} is finished'.format(i))
print('Mean square error: %.4f' % np.mean(np.array(error)**2))
