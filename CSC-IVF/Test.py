# -*- coding: utf-8 -*-
# =============================================================================
# 测试
# =============================================================================
import numpy as np
import torch,os
from PIL import Image
import scipy.io as scio
from skimage.io import imsave
from fusion_layer import l1_addition,saliency_weight
from Net import CsrNet,Decoder

from Arguments import args
device='cuda'
# =============================================================================
# 加载模型
# =============================================================================
LISTA_Detail_Test=CsrNet(
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
        ).to(device)
LISTA_Base_Test=CsrNet(
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
        ).to(device)
LISTA_Decoder_Test=Decoder().to(device)

LISTA_Detail_Test.load_state_dict(torch.load(
         "Train_result\LISTA_Detail_Train.model"
        ))
LISTA_Base_Test.load_state_dict(torch.load(
        "Train_result\LISTA_Base_Train.model"
        ))
LISTA_Decoder_Test.load_state_dict(torch.load(
        "Train_result\LISTA_Decoder.model"
        ))
LISTA_Detail_Test.eval()
LISTA_Base_Test.eval()
LISTA_Decoder_Test.eval()
# =============================================================================
# 测试细节 
# =============================================================================
Test_data_choose='Test_data_FLIR'
#'Test_data_TNO'&'Test_data_NIR_Country'&'Test_data_FLIR'

if Test_data_choose=='Test_data_TNO':
    test_data_path = '.\\Test_data_TNO'
elif Test_data_choose=='Test_data_NIR_Country':
    test_data_path = '.\\Test_data_NIR_Country'
elif Test_data_choose=='Test_data_FLIR':
    test_data_path = '.\\Test_data_FLIR\\'    
# =============================================================================
# 测试开始     
# =============================================================================
def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def fusion_layer(addition_mode,IR_img,VIS_img,alpha=2):
    if addition_mode=='Sum':   
        output = IR_img+VIS_img
        return alpha*output
    elif addition_mode=='Average':
        output = (IR_img+VIS_img)/2   
        return alpha*output,0.5,0.5
    elif addition_mode=='l1_atten':
        output,w_v,w_i = l1_addition(IR_img,VIS_img)
        return alpha*output,w_i,w_v
    elif addition_mode=='saliency_weight':
        output,w_v,w_i = saliency_weight(IR_img,VIS_img)        
        return alpha*output,w_i,w_v

    
    
    
def Test_fusion(img_test1,img_test2):
    img_test1 = np.array(img_test1, dtype='float32')/255# 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    
    img_test2 = np.array(img_test2, dtype='float32')/255 # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
     
    img_test1=img_test1.cuda()
    img_test2=img_test2.cuda()
 

    with torch.no_grad():
        Csc_D_IR=LISTA_Detail_Test(img_test1,'detail')
        Csc_B_IR=LISTA_Base_Test(img_test1,'base')
        Csc_D_VIS=LISTA_Detail_Test(img_test2,'detail')
        Csc_B_VIS=LISTA_Base_Test(img_test2,'base')
    #'Average'&'l1_atten' & saliency_weight  
    base_feature,w_i,w_v=fusion_layer('saliency_weight',Csc_B_IR,Csc_B_VIS,2)    
    detail_feature,w_i,w_v=fusion_layer('l1_atten',Csc_D_IR,Csc_D_VIS,2)

    with torch.no_grad():
        Out=LISTA_Decoder_Test(base_feature,detail_feature)
    return output_img(Out) #输出340*512numpy矩阵 

#确定文件个数
Test_Image_Number=len(os.listdir(test_data_path))
#提取对应的VISi和IRi输入进函数
#得到融合Fi
Figure=[]
Name=[]
R={}

for i in range(int(Test_Image_Number/2)):
    if Test_data_choose=='Test_data_TNO':
        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.bmp') # 读入一张红外的图片
        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.bmp') # 读入一张可见光图的图片
    elif Test_data_choose=='Test_data_NIR_Country':
        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.png') # 读入一张红外的图片
        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.png') # 读入一张可见光图的图片
    elif Test_data_choose=='Test_data_FLIR':
        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.jpg') # 读入一张红外的图片
        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.jpg') # 读入一张可见光图的图片
    Fusion_image=Test_fusion(Test_IR,Test_Vis)
    Figure.append(Fusion_image)
    Name.append('F'+str(i+1))
    Fusion_image_uint8 = (255 * Fusion_image).astype(np.uint8)
    imsave('.\Test_result\F'+str(i+1)+'.png',Fusion_image_uint8)
    R[Name[i]]=Figure[i]   
scio.savemat('Test_result\\R.mat',R)