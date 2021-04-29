#!/usr/bin/env python3
##############################################################################
#################################### Libs ####################################
##############################################################################
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

##############################################################################
################################### Class ####################################
##############################################################################
class Tex_net_mini_block(nn.Module):
    def __init__(self, in_ch, out_ch, k, relu=True):
        super(Tex_net_mini_block, self).__init__()
        self.relu = relu
        
        self.conv = nn.Conv2d(in_ch,  out_ch, k, stride=1, padding=int((k-1)/2))        
        self.norm = nn.BatchNorm2d(out_ch)
        if(self.relu):
            self.ReLU = nn.ReLU()
            
        self.conv.weight = nn.init.xavier_normal_(self.conv.weight)
        self.conv.requires_grad = True
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        
        if(self.relu):
            out = self.ReLU(out)
            
        return out

class Tex_net_block(nn.Module):
    def __init__(self, in_ch, out_ch, UPSAMPLE=True):
        super(Tex_net_block, self).__init__()
        
        if(UPSAMPLE):
            self.conv  = nn.Sequential(Tex_net_mini_block(in_ch, out_ch,  3), 
                                       Tex_net_mini_block(out_ch, out_ch, 3), 
                                       Tex_net_mini_block(out_ch, out_ch, 1, False),
                                       nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.ReLU())
        else:
            self.conv  = nn.Sequential(Tex_net_mini_block(in_ch, out_ch,  3), 
                                       Tex_net_mini_block(out_ch, out_ch, 3), 
                                       Tex_net_mini_block(out_ch, out_ch, 1))
        
        self.conv.requires_grad = True

    def forward(self, x):
        out = self.conv(x)
        return out
    
class rand_head(nn.Module):
    def __init__(self, M, BATCH=1, device='cuda'):
        super(rand_head, self).__init__()
        self.M = M
        self.device = device
        self.BATCH = BATCH
        
        self.conv = Tex_net_block(3, 3, False)
        self.conv.requires_grad = True
    
    def forward(self):
        out = torch.randn((self.BATCH, 3, self.M, self.M)).to(self.device)
        out = self.conv(out)
        return out
        

class TexNet(nn.Module):
    def __init__(self, M=256, K=5, BATCH=1, descriptor=None, device='cuda'):
        super(TexNet, self).__init__()
        self.M = M
        self.K = K
        self.descriptor = descriptor
        self.device = device
        self.norm_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        self.first_up = nn.Upsample(scale_factor=2, mode='bilinear').to(self.device)
        
        self.to_image = nn.Conv2d(3*self.K, 3, 1, stride=1, padding=0).to(self.device)
        self.to_image.weight = nn.init.xavier_normal_(self.to_image.weight)
        
        in_ch  = 6
        self.norm    = []
        self.body_op = []
        self.head_op = []
        self.M_i     = []
        for i in range(self.K):
            self.M_i.append( int( self.M / (2**(self.K-i-1)) ) )
            self.head_op.append( rand_head(self.M_i[-1], BATCH=BATCH, device=self.device).to(self.device) )
            
        for i in range(1, self.K):
            if(i == self.K-1):
                self.body_op.append( Tex_net_block(in_ch, in_ch, False).to(self.device) )
            else:
                self.body_op.append( Tex_net_block(in_ch, in_ch).to(self.device) )
            self.norm.append( nn.BatchNorm2d(in_ch).to(self.device) )
            in_ch += 3
            

    def forward(self):
        out   = self.first_up( self.head_op[0]() )
        
        for i in range(len(self.body_op)):
            noise = self.head_op[i+1]()
            out   = self.body_op[i](self.norm[i]( torch.cat((out, noise),1) ))
        
        out = self.to_image(out)
        #for i in range(out.shape[0]):
        #    out[i] = self.norm_image(out[i])
            
        if(self.descriptor != None):
            self.descriptor.eval()
            out = self.descriptor(out)
        #else:
        #    out[:,0,:,:] =  out[:,0,:,:]*0.229 + 0.485
        #    out[:,1,:,:] =  out[:,1,:,:]*0.224 + 0.456
        #    out[:,2,:,:] =  out[:,2,:,:]*0.225 + 0.406
                        
        return out


