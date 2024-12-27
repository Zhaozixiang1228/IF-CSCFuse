# -*- coding: utf-8 -*-

import torchvision
from torchvision import transforms
import torchvision.utils as vutils     
import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
from Net import CsrNet,Decoder
from Arguments import args
time_start=time.time()

Train_Image_Number=len(os.listdir(args.train_data_path+'train2014\\'))#确定训练图片个数
Iter_per_epoch=(Train_Image_Number % args.batch_size!=0)+Train_Image_Number//args.batch_size
# =============================================================================
# 预处理和数据集建立 
# =============================================================================

transforms = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(128),
        transforms.RandomResizedCrop(args.img_size),#分别对其进行随机大小和随机宽高比的裁剪，之后resize到指定大小256
        #transforms.RandomHorizontalFlip(),#对原始图像进行随机的水平翻转
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])
                          
Data = torchvision.datasets.ImageFolder(args.train_data_path,transform=transforms)
dataloader = torch.utils.data.DataLoader(Data, args.batch_size,shuffle=True)

# =============================================================================
# 模型
# =============================================================================
LISTA_Detail_Train=CsrNet(
        img_channels=args.num_input_channels, 
        in_channels=args.kernel_channel, 
        out_channels=args.kernel_channel, 
        kernel_size=args.kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=False, 
        ista_iters=args.ista_iters,# 迭代次数
        act='sst', # activation function (None will disable it)
        act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
        norm=True, # batch_normalization (None will disable it)
        padding_mode='reflection',
        padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
        )
LISTA_Base_Train=CsrNet(
        img_channels=args.num_input_channels, 
        in_channels=args.kernel_channel, 
        out_channels=args.kernel_channel, 
        kernel_size=args.kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=False, 
        ista_iters=args.ista_iters,# 迭代次数
        act='prelu', # activation function (None will disable it)
        act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
        norm=True, # batch_normalization (None will disable it)
        padding_mode='reflection',
        padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
        )
LISTA_Decoder=Decoder()
if args.is_cuda:
    LISTA_Detail_Train=LISTA_Detail_Train.cuda()
    LISTA_Base_Train=LISTA_Base_Train.cuda()
    LISTA_Decoder=LISTA_Decoder.cuda()
print(LISTA_Detail_Train)
print(LISTA_Base_Train)
print(LISTA_Decoder)

optimizer1 = optim.Adam(LISTA_Detail_Train.parameters(), lr = args.lr)
optimizer2 = optim.Adam(LISTA_Base_Train.parameters(), lr = args.lr)
optimizer3 = optim.Adam(LISTA_Decoder.parameters(), lr = args.lr)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer1, [args.epochs//2+0.1, args.epochs+0.1], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer2, [args.epochs//2+0.1, args.epochs+0.1], gamma=0.1)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer3, [args.epochs//2+0.1, args.epochs+0.1], gamma=0.1)

MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
SSIMLoss = kornia.losses.SSIM(3, reduction='mean')

# =============================================================================
# 训练
# =============================================================================
print('============ Training Begins ===============')
print('The total number of images is %d,\n Need to cycle %d times.'%(Train_Image_Number,Iter_per_epoch))

loss_train=[]
loss_mse=[]
loss_ssim=[]
lr_list=[]
theta_Encoder_Detail=[]

for iteration in range(args.epochs):


    LISTA_Detail_Train.train()
    LISTA_Base_Train.train()
    LISTA_Decoder.train()
    
    data_iter_input = iter(dataloader)
    
    for step in range(Iter_per_epoch):
        img_input,_ =next(data_iter_input)
        
          
        if args.is_cuda:
            img_input=img_input.cuda()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        # =====================================================================
        # 计算损失 
        # =====================================================================

        #输入网络 
        csc_D=LISTA_Detail_Train(img_input,'detail')
        csc_B=LISTA_Base_Train(img_input,'base')
        img_recon=LISTA_Decoder(csc_B,csc_D)
        
#        #MSELoss,SmoothL1Loss,L1Loss 损失 
        mse_recon=MSELoss(img_input,img_recon)
        ssim_recon=SSIMLoss(img_input,img_recon)
      
        loss = mse_recon + 5*ssim_recon

        
        #更新模型参数
        loss.backward()
        optimizer1.step() 
        optimizer2.step()
        optimizer3.step()
        # =====================================================================
        # 输出结果 
        # =====================================================================
        los = loss.item()
        mse_r=mse_recon.item()
        ssim_r=ssim_recon.item()

        if (step + 1) % Iter_per_epoch == 0:
            print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, step+1, los, optimizer1.state_dict()['param_groups'][0]['lr']))
            print('Total:MSELoss:%.7f,SSIMLoss:%.7f'%(mse_r,ssim_r))

          
        # =====================================================================
        # 储存数据
        # =====================================================================
        #Save Loss
        loss_train.append(los)
        loss_mse.append(mse_r)
        loss_ssim.append(ssim_r)

    scheduler1.step()#更新学习率 
    scheduler2.step()
    scheduler3.step()
    lr_list.append(optimizer1.state_dict()['param_groups'][0]['lr'])

def Average_loss(loss):
    return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]

plt.figure(figsize=[16,4])
plt.subplot(131)
plt.plot(Average_loss(loss_train)), plt.title('Loss')
plt.subplot(132)
plt.plot(Average_loss(loss_mse)), plt.title('loss_mse')
plt.subplot(133)
plt.plot(Average_loss(loss_ssim)), plt.title('loss_ssim')
plt.tight_layout()
plt.savefig(os.path.join(args.train_path,'curve_per_epoch.png'))    

# save model
LISTA_Detail_Train.eval()
LISTA_Detail_Train.cpu()
LISTA_Base_Train.eval()
LISTA_Base_Train.cpu()
LISTA_Decoder.eval()
LISTA_Decoder.cpu()

name=[
      'LISTA_Detail_Train.model',
      'LISTA_Base_Train.model',
      'LISTA_Decoder.model',
      ]
for i in range(4):
    save_model_filename =str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" +name[i] 
    save_model_path = os.path.join(args.train_path, save_model_filename)
    if i == 0:
        torch.save(LISTA_Detail_Train.state_dict(), save_model_path)
    elif i == 1:
        torch.save(LISTA_Base_Train.state_dict(), save_model_path)
    elif i == 2:
        torch.save(LISTA_Decoder.state_dict(), save_model_path)        
    print("\nDone, trained model saved at", save_model_path)
time_end=time.time()
print('time cost %.2f s'%(time_end-time_start))
