# !pip install torch==1.5.0 torchvision==0.6.0
from tools.imports import *

content_im = im.load('images/malmoe.jpg')
style_im = im.load('images/starry_night.jpg')
content_layers = ['conv_4_2']
im_size = 512 if torch.cuda.is_available() else 128

style_layers = nt.layers
[style_layers.remove(i) for i in content_layers]
net = nt.Styler(content_im, style_im, im_size, content_layers, style_layers)
# Use net.summary() to see layers.

# net.summary()
image = run.run_styler(net, 300, 1e-4)