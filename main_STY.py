#!/usr/bin/env python3
##############################################################################
#################################### Libs ####################################
##############################################################################
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision

import time

import tools.net as nt
import tools.image as im 
from tools.run import run_styler 
from tools.texnet import *


def save_model(net, fn):
    outfile = open(fn,'wb')
    pickle.dump(net,outfile)
    outfile.close()
    return

def load_model(fn):
    infile = open(fn,'rb')
    net = pickle.load(infile)
    infile.close()
    return net


##############################################################################
#################################### Vars ####################################
##############################################################################
### Check for GPU ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Flags ###
TRAIN_TEXNET = True
SAVE_MODEL = True

### Normalizeation function ###
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

### Image size of the output ###
im_size = 128 
BATCH   = 1

### Context and Stlye paths ###
s_path = '../textures/'
c_path = '../context/'

s_fn = 'starry_night_small.jpg'
#s_fn = 'wave_small.png'
c_fn = 'malmoe_small.jpg'

### Load in the images ###
content_im = im.load(c_path + c_fn, im_size)
style_im = im.load(s_path + s_fn, im_size)
temp_c = deepcopy(content_im)
temp_s = deepcopy(style_im)
for i in range(1,BATCH):
    temp_c = torch.cat((temp_c, content_im),0)
    temp_s = torch.cat((temp_s, style_im),0)
content_im = temp_c
style_im = temp_s
    
### Choose which layers to enforce the context loss ###
content_layers = ['relu_4_2']

### Choose which layers to enforce the style loss ###
style_layers = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']

### Define the network ###
transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        normalize,
        ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH,
                              shuffle=True, num_workers=2)
VGG = nt.Styler(content_im, style_im, im_size, content_layers, style_layers, random=True, batch=BATCH, size=im_size, loader=loader)

### Which dataset to look at ###
#texture_path = 'textures/'
#texture_fn   = 'wave_small.png'
#texture      = normalize(utils.load_image_as_tensor(texture_path + texture_fn, VERBOSE=True))

### Save Params ###
TextureNet_name = 'Style_Net_'

### Configuration ###
MAX_ITERATIONS = 200
STEPS          = (np.array([0.5, 0.6, 0.7, 0.8, 0.9]) * MAX_ITERATIONS).astype(int)
LR_DECAY       = 0.7
LEARNING_RATE  = 0.10
PROGRESS_RATE  = 1
K              = 6
LOSS_RATIO     = 1e-7

### Setup the model ###    
MODEL_PATH     = '../models/'    

    
tnet = TexNet(M=im_size, K=5, BATCH=BATCH, descriptor=None)
tnet = tnet.to(device)

h_params = [h.parameters for h in tnet.head_op]
b_params = [b.parameters for b in tnet.body_op]
a = tnet.parameters
optimizer = optim.Adam([{'params': tnet.parameters()},
                        {'params': h_params[0]()},
                        {'params': h_params[1]()},
                        {'params': h_params[2]()},
                        {'params': h_params[3]()},
                        {'params': h_params[4]()},
                        {'params': b_params[0]()},
                        {'params': b_params[1]()},
                        {'params': b_params[2]()},
                        {'params': b_params[3]()}],
                       lr=LEARNING_RATE)

tnet.descriptor = VGG.to(device)
tnet.descriptor.requires_grad = False
#criterion = nn.CrossEntropyLoss()

##############################################################################
#################################### Main ####################################
##############################################################################
#tnet = load_model(MODEL_PATH + TextureNet_name + s_fn.split('.')[0] + '.pkl')
if(TRAIN_TEXNET):            
    def closure():
        optimizer.zero_grad()
        trash = tnet()

        content_loss = 0
        style_loss = 0

        for c in tnet.descriptor.content_losses: content_loss += c.loss
        for s in tnet.descriptor.style_losses: style_loss += s.loss

        if(style_loss > 1e3):
            print('Overflow, style_loss = %.3f' % (style_loss))
            return np.inf
            
        if(torch.max(style_loss/LOSS_RATIO) > 100):
            pass
        else:
            style_loss /= LOSS_RATIO
        loss = (content_loss + style_loss)/len(tnet.descriptor.style_losses)
        loss = style_loss/len(tnet.descriptor.style_losses)
                    
        #print("Content loss: {}, style loss: {}".format(content_loss, style_loss))
        loss.backward()
                
        return loss
        
    t_start = time.time()
    BEST_LOSS = np.inf
    for i in range(MAX_ITERATIONS):
        l = optimizer.step(closure)
        #print('Epoch: %d \t Loss: %.3f' % (i+1, l))
        
        if(i+1 in STEPS):
            LEARNING_RATE *= LR_DECAY
            optimizer.param_groups[-1]['lr'] = LEARNING_RATE
              
        if( ((i+1) % PROGRESS_RATE) == 0 ):
            p = ((i+1)/MAX_ITERATIONS)
            c_t = (time.time() - t_start)/60
            print('Progress = %d / %d (%0.2f%%) \t Time = %.2f m \t Est Time %.2f m \t lr = %.4f m \t Loss: %.10f'%(\
                  i+1, MAX_ITERATIONS, p*100, c_t, c_t/p, LEARNING_RATE, l))
                
        if(l > 2*BEST_LOSS):
            continue
        elif(l < BEST_LOSS):
            BEST_LOSS = l

    if(SAVE_MODEL):
        print('Saving...')
        save_model(tnet, MODEL_PATH + TextureNet_name + s_fn.split('.')[0] + '.pkl')

torch.cuda.empty_cache()



### Display Texture ###
tnet.descriptor = None
gen_texture = tnet()
for i in range(BATCH):
    
    ### Plot ###
    fig, ax = plt.subplots(1, 3, figsize=[8,4])
    im.unload(style_im[i], ax[0])
    im.unload(content_im[i], ax[1])
    im.unload(gen_texture[i], ax[2])
    
    








    
