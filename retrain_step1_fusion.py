#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:59:02 2020

@author: weizhe
"""
from CNNs import *
from utils import *
import os

preT_folder='ckpt/step1'
scale=[256,128,64,32]
ckpt=preT_folder

model=Fusion()
# Load pretrained scales
load_dict={}
for s in scale:
    pre='model{}.'.format(s)
    stat=torch.load(os.path.join(preT_folder,'{}.pth'.format(s)))
    preT_dict=stat['model']
    D={pre+k: v for k, v in preT_dict.items() if '5' not in k and '6' not in k}
    load_dict.update(D)
model_dict=model.state_dict()
model_dict.update(load_dict)
model.load_state_dict(model_dict)
print('Successfully load pretrained model.')
model.to(torch.device('cuda'))

# Frozen
# for param in model.model256.parameters():
#     param.requires_grad = False
# for param in model.model128.parameters():
#     param.requires_grad = False
# for param in model.model64.parameters():
#     param.requires_grad = False
# for param in model.model32.parameters():
#     param.requires_grad = False

# Train
loss_cn=nn.BCEWithLogitsLoss()
loss_rn=Regularization(model, 1.e-7, p=1)
opt=torch.optim.Adam(model.parameters(), lr=1.0e-4)
Steps=30000
trainGener=gen_multiscale('/media2/Data/deblurQA/Step1_train_.txt')
valGener=gen_multiscale('/media2/Data/deblurQA/Step1_val_.txt')
for i in range(Steps):
    batch_X,batch_Y = trainGener.__next__()
    X=batch_X
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
    if i%10==0:
        print('[Train]: Step: %d, Loss: %.4f, Class loss: %.4f' % (i, loss.data, loss_c.data))
    # if i % 50 ==0:
    #     batch_X,batch_Y = valGener.__next__()
    #     X=batch_X
    #     Y=batch_Y.cuda()
    #     model.eval()
    #     Y_=model(X)
    #     logits=torch.sigmoid(Y_.detach())>0.5
    #     correct=torch.eq(logits, Y)
    #     rate=correct.float().mean()
    #     print('[Validate]: Step: %d, Correct: %.4f' % (i, rate.data))

# Save
torch.save(model.state_dict(), os.path.join(ckpt,'fusion_classify_.pth'))
print('Model saved at %s' % ckpt)
