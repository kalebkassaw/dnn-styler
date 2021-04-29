#!/usr/bin/env python3
##############################################################################
#################################### Libs ####################################
##############################################################################
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
import torch

import time

##############################################################################
#################################### Func ####################################
##############################################################################
def load_image_as_tensor(fn, VERBOSE=False):
    texture      = Image.open(fn)

    ### Display the texture used ###
    if(VERBOSE):
        plt.imshow(texture)
        plt.axis('off')
        plt.show()
        
    ### Convert to torch tensor ###
    texture = asarray(texture)
    texture = np.swapaxes(texture, 0,1)
    texture = np.swapaxes(texture, 0,2)
    
    ### Cast to float between 0 and 1 ###
    texture = texture.astype(np.float32)
    texture = texture / 255
    
    texture = torch.from_numpy(texture)
    
    return texture
