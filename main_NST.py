#!/usr/bin/env python3
##############################################################################
#################################### Libs ####################################
##############################################################################
import torch
from tools.imports import *
from copy import deepcopy    

##############################################################################
#################################### Vars ####################################
##############################################################################
layer_dict = {
        'conv_1_1':  0, 'conv_1_2':  2,
        'conv_2_1':  5, 'conv_2_2':  7,
        'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16,
        'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25,
        'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34,
        }

layers = ['conv_1_1', 'conv_1_2',
          'conv_2_1', 'conv_2_2',
          'conv_3_1', 'conv_3_2', 'conv_3_3', 'conv_3_4',
          'conv_4_1', 'conv_4_2', 'conv_4_3', 'conv_4_4',
          'conv_5_1', 'conv_5_2', 'conv_5_3', 'conv_5_4']

### Check for GPU ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

### Image size of the output ###
im_size = 400 

### Transfer Params ###
EPOCHS = 200
LOSS_RATIO = 1e-1

### Context and Stlye paths ###
s_path = '../textures/'
c_path = '../context/'

s_fn = 'starry_night_8x6.jpg'
c_fn = '../context/malmoe_small.jpg'
c_fn = '../context/Tuebingen_Neckarfront.jpg'

### Load in the images ###
content_im = im.load(c_path + c_fn, im_size)
style_im = im.load(s_path + s_fn, im_size)

### Choose which layers to enforce the context loss ###
content_layers = ['conv_4_2']

### Choose which layers to enforce the style loss ###
style_layers = ['conv_3_1', 'conv_4_1', 'conv_5_1']


### Define the network ###
net = nt.Styler(content_im, style_im, im_size, content_layers, style_layers, layer_dict, layers)
##############################################################################
#################################### Main ####################################
##############################################################################
### Clear the cache ###
torch.cuda.empty_cache()

### Run the optimization ###
net.eval()
x = deepcopy(content_im)
#x = torch.randn(content_im.shape)
nst = run_styler(net, x, EPOCHS, LOSS_RATIO, device)

### Clear the cache ###
torch.cuda.empty_cache()

### Plot ###
fig, ax = plt.subplots(3, 1, figsize=[8,24])
im.unload(style_im, ax[0])
im.unload(content_im, ax[1])
im.unload(nst, ax[2])













    
