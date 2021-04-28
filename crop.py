# crop.py
from PIL import Image

images = []#['assise', 'shipwreck', 'scream', 'sunflowers', 'malmoe', 'nyhavn', 'london', 'flower', 'bird', 'aarhus', 'starry_night', 'wave']
verticals = ['flower', 'sunflowers', 'assise', 'scream']

for ig in images:
    with Image.open('images/orig/' + ig + '.jpg') as i:
        i_size = i.size
        print(ig, i_size)
        if i_size[0] > i_size[1]:
            width = int(512 * i_size[0] / i_size[1])
            i = i.resize([width, 512])
            cutwidth = (width - 512) / 2
            cutwidth += 30 if ig == 'london' else 0
            i = i.crop([cutwidth, 0, cutwidth+512, 512])
            if ig in verticals: i = i.rotate(270)
        else:
            height = int(512 * i_size[1] / i_size[0])
            i = i.resize([512, height])
            cutheight = (height - 512) / 2
            i = i.crop([0, cutheight, 512, cutheight+512])
        i = i.resize([512, 512])
        i.save('images/' + ig + '.jpg')