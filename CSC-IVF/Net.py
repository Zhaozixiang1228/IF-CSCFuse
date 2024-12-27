# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np


Blur = kornia.filters.BoxBlur((11, 11))#定义低通滤波初始化   

device='cuda'

class SST(nn.Module):
    def __init__(self, num_parameters=1, init=1e-2):
        super(SST,self).__init__()
        self.num_parameters=num_parameters
        self.init=init
        self.theta = nn.Parameter(torch.full(size=(1, self.num_parameters, 1, 1), 
                                             fill_value=self.init)) 
        self.relu = nn.ReLU(True)        
    def forward(self, x):
        output=x.sign()*self.relu(x.abs()-self.theta)
        return output

def get_activation(act='relu', 
                   num_parameters=None, 
                   init=None):
    if act.lower() not in ['relu','prelu','sst']:
        raise Exception('Only support "relu","prelu" or "sst". But get "%s."'%(act))
    if act.lower()=='relu':
        return nn.ReLU(True)
    if act.lower()=='prelu':
        num_parameters=1 if num_parameters==None else num_parameters
        init=0.25 if init==None else init
        return nn.PReLU(num_parameters, init)
    if act.lower()=='sst':
        num_parameters=1 if num_parameters==None else num_parameters
        init=1e-2 if init==None else init
        return SST(num_parameters, init)

def get_padder(padding_mode='reflection',
               padding=1,
               value=None):
    if padding_mode.lower() not in ['reflection','replication','zero','zeros','constant']:
        raise Exception('Only support "reflection","replication","zero" or "constant". But get "%s."'%(padding_mode))
    if padding_mode.lower()=='reflection':
        return nn.ReflectionPad2d(padding)
    if padding_mode.lower()=='replication':
        return nn.ReplicationPad2d(padding)
    if padding_mode.lower() in ['zero','zeros']:
        return nn.ZeroPad2d(padding)
    if padding_mode.lower() in 'constant':
        value=0 if value==None else value
        return nn.ConstantPad2d(padding,value)

class Conv2d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride,
                 dilation, 
                 groups, 
                 bias,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(Conv2d, self).__init__()
        padding = int(int(1+dilation*(kernel_size-1))//2) if padding=='same' else 0
        self.pad = get_padder(padding_mode, padding, value)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                0, dilation, groups, bias)
    def forward(self, code):
        return self.conv2d(self.pad(code))

    
class DictConv2d(nn.Module):
    def __init__(self, img_channels, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, # only support stride=1
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 num_convs=1,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(DictConv2d, self).__init__()
        self.in_channels = in_channels
        
        
        # decoder
        self.conv_decoder = nn.Sequential()
        self.conv_decoder.add_module('de_conv0', Conv2d(in_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value) )
        for i in range(1,num_convs):
            self.conv_decoder.add_module('de_conv'+str(i), Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value))

        # encoder
        self.conv_encoder = nn.Sequential()
        for i in range(num_convs-1):
            self.conv_encoder.add_module('en_conv'+str(i), Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value))
        self.conv_encoder.add_module('en_conv'+str(num_convs-1), Conv2d(img_channels, in_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value) )

        
        # 如果输入输出通道数不一致，使用1x1卷积改变conv_encoder的通道数
        self.shift_flag = out_channels != in_channels
        self.conv_channel_shift = nn.Conv2d(in_channels, out_channels, 1) if self.shift_flag else None
            
    def _forward(self, data, code):
        B,_,H,W = data.shape
        code = torch.zeros(B,self.in_channels,H,W).cuda() if code is None else code 
#        code = torch.zeros(B,self.in_channels,H,W).to(data.device).to(data.dtype) if code==None else code         
        dcode = self.conv_decoder(code)
        
        if data.shape[2]!=dcode.shape[2] or data.shape[3]!=dcode.shape[3]:
            data = F.interpolate(data, size=dcode.shape[2:])#插值或采样 
        res = data - dcode
        dres = self.conv_encoder(res)
        
        if code.shape[2]!=dres.shape[2] or code.shape[3]!=dres.shape[3]:
            code = F.interpolate(code, size=dres.shape[2:])
        code = code+dres
        
        if self.shift_flag:
            code = self.conv_channel_shift(code)
        return code
    
    def forward(self, data, code):
        return self._forward(data, code)

debug=False  # True
if debug:
    img_channels=3
    in_channels=64
    out_channels=64
    kernel_size=3
    batch=2
    height, width = 32,32
    f = DictConv2d(img_channels, in_channels, out_channels, kernel_size)
    data = torch.rand(batch, img_channels, height, width)
    code = None
    out1 = f(data,code)
    code = torch.rand(batch, in_channels, height, width)
    out2 = f(data,code)
    print([out1.shape,out2.shape])

class DictConv2dBlock(nn.Module):
    def __init__(self, img_channels, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 num_convs=1, # the number of conv units in decoder and encoder
                 act='sst', # activation function (None will disable it)
                 act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
                 norm=True, # batch_normalization (None will disable it)
                 padding_mode='reflection',
                 padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
                 ):
        super(DictConv2dBlock, self).__init__()
        self.conv = DictConv2d(img_channels,in_channels,out_channels, kernel_size, stride, 
                   dilation, groups, bias, num_convs)
        self.norm = nn.BatchNorm2d(out_channels) if norm!=None else None
        num_parameters = out_channels if act!='relu' else None
        self.activation = get_activation(act, num_parameters, act_value) if act!=None else act
        self.act=act
    def _forward(self, data, code):
        code = self.conv(data,code)
        if self.norm!=None:
            code = self.norm(code)
        if self.activation!=None:

            code = self.activation(code)
            return data, code           
    
    def forward(self, data, code):
        return self._forward(data, code)

debug=False
if debug:
    img_channels=3
    in_channels=64
    out_channels=64
    kernel_size=3
    batch=2
    height, width = 32,32
    f = DictConv2dBlock(img_channels, in_channels, out_channels, kernel_size)
    data = torch.rand(batch, img_channels, height, width)
    code = None
    out1 = f(data,code)
    code = torch.rand(batch, in_channels, height, width)
    out2 = f(data,code)
    print([out1.shape,out2.shape])
    


class BlurConv2d(nn.Module):
    def __init__(self, kernel_size=11):
        super(BlurConv2d, self).__init__()
        self.kernel = nn.Parameter(kornia.filters.get_box_kernel2d([kernel_size,kernel_size]))
        self.border_type = 'reflect'
        
    def _forward(self, input):
        return kornia.filter2D(input, self.kernel, self.border_type)
        
    def forward(self, input):
        return self._forward(input)        
    
class CsrNet(nn.Module):
    def __init__(self, img_channels, 
        in_channels, 
        out_channels, 
        kernel_size, 
        ista_iters,
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=False, 
        num_convs=1, # the number of conv units in decoder and encoder
        act='sst', # activation function (None will disable it)
        act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
        norm=True, # batch_normalization (None will disable it)
        padding_mode='reflection',
        padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
        ):
        super(CsrNet, self).__init__()   
        self.convLayer0 = DictConv2dBlock(
                img_channels,in_channels,out_channels, kernel_size, stride, 
                dilation, groups, bias, num_convs,act,act_value) 
        self.convLayer = nn.ModuleList(
                [DictConv2dBlock(
                        img_channels,in_channels,out_channels, kernel_size, stride, 
                        dilation, groups, bias, num_convs,act,act_value) 
                for i in range(ista_iters)])
        self.BlurFilter = BlurConv2d
  
        self.act=act
        
    def forward(self,img,model_class):
        if model_class.lower()=='base':
            img_input=Blur(img)
        elif model_class.lower()=='detail':
            img_input=img-Blur(img)
        
        #初始化z0
        img_blur,z_k=self.convLayer0(img_input, code = None)
        for layer in self.convLayer:
            img,z_k = layer(img,z_k)    
        return z_k
        
        
class Decoder(nn.Module):
    def __init__(self, kernel_size=11,in_channels=64):
        super(Decoder, self).__init__()        
        self.decode_last=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels//2, 3, padding=0, bias=False), 
            nn.BatchNorm2d(in_channels//2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels//2, in_channels//4, 3, padding=0, bias=False), 
            nn.BatchNorm2d(in_channels//4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels//4, 1, 3, padding=0, bias=False), 
            nn.BatchNorm2d(1),
            )  
    def forward(self, Cn_b, Cn_d):
        Output_b=self.decode_last(Cn_b)
        Output_d=self.decode_last(Cn_d)
        Cn=Output_b+Output_d
        Output=torch.sigmoid(Cn)
        return Output
    
    