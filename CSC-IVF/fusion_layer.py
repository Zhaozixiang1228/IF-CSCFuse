# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
device='cuda'
def l1_addition(y1,y2,window_width=1):
    ActivityMap1 = y1.abs()
    ActivityMap2 = y2.abs()
    spatial_map1 = ActivityMap1.sum(dim=1, keepdim=True)
    spatial_map2 = ActivityMap2.sum(dim=1, keepdim=True)

    kernel = torch.ones(1,1,2*window_width+1,2*window_width+1).type(torch.float32)/((2*window_width+1)**2)
    kernel=kernel.to(device)

    ActMap1 = F.conv2d(spatial_map1, kernel, padding=window_width)
    ActMap2 = F.conv2d(spatial_map2, kernel, padding=window_width)
    WeightMap1 = torch.div(ActMap1,(ActMap1+ActMap2))
    WeightMap2 = torch.div(ActMap2,(ActMap1+ActMap2))
    Fusion_result = WeightMap1.mul(y1)+WeightMap2.mul(y2)
    return Fusion_result,WeightMap1,WeightMap2
  



# addition fusion strategy
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f,spatial_w1,spatial_w2= f_spatial
    return tensor_f,spatial_w1,spatial_w2


def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + 1e-10)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + 1e-10)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f,spatial_w1,spatial_w2


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

# =============================================================================
# 注意力加权 
# =============================================================================

from scipy.spatial.distance import cdist
import torch.nn
from cv2.ximgproc import guidedFilter
import cv2

def SalWeights(G):
    N = len(G)
    W = []
    
    for i in range(N):
        W.append(saliency(G[i]))

    W = np.dstack(W) + 1e-12
    W = W / W.sum(axis=2, keepdims=True)
    return W

D = cdist(np.arange(256)[:,None], np.arange(256)[:,None], 'euclidean')
def saliency(img):
    global D
    hist = np.bincount(img.flatten(), minlength=img.max()-img.min()) / img.size
    sal_tab = np.dot(hist, D)
    z = sal_tab[img]
    return z

def FuseWeights(G, W):
    return np.sum(np.dstack(G)*W, axis=2)
def Guided(I, p, r, eps):
    return guidedFilter(I, p, r, eps)
def GuidedOptimize(G, P, r, eps):
    N = len(G)
    W = []
    
    for i in range(N):
        # MOST COSTLY OPERATION IN THE WHOLE THING
        W.append(Guided(G[i].astype(np.float32), P[i].astype(np.float32), r, eps))
    
    W = np.dstack(W) + 1e-12
    W = W / W.sum(axis=2, keepdims=True)
    return W
def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]
def saliency_weight(ir,vis):
    #输入[1,1,H,W]的特征图（0~1） 
    w_ir=torch.ones(ir.shape,dtype=torch.float).cuda()
    w_vis=torch.ones(vis.shape,dtype=torch.float).cuda()
    r1 = 45
    eps1 =  0.01
    for i in range(ir.shape[1]):
        vis_img = vis[:,i,:,:].cpu().detach().numpy()[0,:,:]
        ir_img  = ir[:,i,:,:].cpu().detach().numpy()[0,:,:]  
        vis_img_norm=(vis_img-vis_img.min())/(vis_img.max()-vis_img.min())
        ir_img_norm=(ir_img-ir_img.min())/(ir_img.max()-ir_img.min())
        
        p1_0 = SalWeights([(vis_img_norm*255).astype(np.int64),
                        (ir_img_norm*255).astype(np.int64)]
                        )
        P1 = [p1_0[:,:,0], 
              p1_0[:,:,1]]
        Wb = GuidedOptimize([vis_img_norm, ir_img_norm], P1, r1, eps1)
        wb_v=Wb[:,:,0]
        wb_i=Wb[:,:,1]
        
        wb_v_weight = torch.FloatTensor(
                wb_v.reshape(1,1,wb_v.shape[0],wb_v.shape[1])).cuda()
        
        wb_i_weight = torch.FloatTensor(
                wb_i.reshape(1,1,wb_i.shape[0],wb_i.shape[1])).cuda()
        
        w_ir[:,i,:,:]=wb_i_weight
        w_vis[:,i,:,:]=wb_v_weight
    fusion=ir.mul(w_ir)+vis.mul(w_vis)
    return fusion,w_ir,w_vis