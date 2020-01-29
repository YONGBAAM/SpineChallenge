import pandas as pd
import numpy as np
import os
import scipy.io as spio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from label_io import read_labels

a = np.array(np.arange(20)/20.0)
a = np.tile(a.reshape(-1,1,1), (1,20,3))

plt.figure()
plt.imshow(a)
plt.scatter(15,2)
plt.show()

plt.figure()
img = Image.open('./resized_images/sunhl-1th-02-Jan-2017-162 A AP.jpg')
img = np.asarray(img)
plt.imshow(img)
label = read_labels(label_location='./resized_labels', relative=True)
label = label[0].reshape(-1,2)
label[:,0] *= 256
label[:,1] *= 512
plt.scatter(label[:,0], label[:,1])
plt.show()