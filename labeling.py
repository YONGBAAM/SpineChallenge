import os
from os.path import join as osj
from PIL import Image

import numpy as np

image_location = './record_images_indexed'


im = Image.open(osj(image_location, '0000.jpg')).convert('RGB')

hsvim = im.convert('HSV')

rgbarr = np.array(im)
hsvarr = np.array(hsvim)
H,W,_ = rgbarr.shape
green_image = np.zeros((H,W))


h = hsvarr[:,:,0]

#angle
lo=100
hi = 140
lo = int(lo/360*255)
hi = int(hi/360*255)

green = np.where((h>lo)&(h <hi))

green_image[green] = 1.0

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(121)
plt.imshow(im)
plt.subplot(122)
plt.imshow(rgbarr[:,:,1])
plt.show()
