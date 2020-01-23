import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
from helpers import read_data_names, read_labels, plot_image, chw, hwc
from dataset import SpineDataset
from model import SegmentNet, LandmarkNet, get_classifier, SpinalStructured
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

    _pil_interpolation_to_str = {
        Image.NEAREST: 'PIL.Image.NEAREST',
        Image.BILINEAR: 'PIL.Image.BILINEAR',
        Image.BICUBIC: 'PIL.Image.BICUBIC',
        Image.LANCZOS: 'PIL.Image.LANCZOS',
        Image.HAMMING: 'PIL.Image.HAMMING',
        Image.BOX: 'PIL.Image.BOX',
    }

##################################
#
#   Define custom transform
#   현재 좌표모드만 제작완료
#   이름 다르게해서 헷갈리지 않게 하기
#    모두 PIL image 라 생각하고!
#   사실 torchvision transform도
#   PIL 이미지에서 작업함!
#
#################################




class CoordCustomPad:
    def __init__(self, HoverW, fixH=True):
        self.ratio = HoverW
        self.fixH = fixH
        self.fixW = not fixH

    def __call__(self, image, label):
        W, H = image.size

        desire_W = int(H / self.ratio)
        left_pad = int((desire_W - W) / 2)
        right_pad = desire_W - W - left_pad
        im_pad = TF.pad(image, padding=(left_pad, 0, right_pad, 0), padding_mode='constant')

        r_label = label.reshape(-1,2)
        r_label[:,0] += left_pad

        return im_pad, r_label

class CoordRandomRotate:
    def __init__(self,max_angle, expand = False, is_random = True):
        self.max_angle = max_angle
        self.expand = expand
        self.is_random = is_random
    def __call__(self, image, label):
        if self.is_random:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
        else:
            angle = self.max_angle

        W1, H1 = image.size

        image = image.rotate(angle, expand = self.expand)
        if self.expand:
            W2, H2 = image.size
            label = rotate_label(label, angle, H = H1, W = W1, new_centerXY=(W2 / 2, H2 / 2))
        else:
            label = rotate_label(label, angle, H = H1, W = W1)
        return image, label

class CoordVerticalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, label):
        W,H = image.size

        if np.random.random() < self.prob:
            r_img = TF.vflip(image)
            label = label.reshape(-1, 2)
            r_label = np.zeros_like(label)
            r_label[:, 0] = label[:, 0]
            r_label[:, 1] = H - 1 - label[:, 1]
            r_label = r_label.reshape(-1)
            return r_img, r_label
        else:
            return image, label
class CoordHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        W, H = image.size

        if np.random.random() < self.prob:
            r_img = TF.hflip(image)
            label = label.reshape(-1, 2)
            r_label = np.zeros_like(label)
            r_label[:,1] = label[:,1]
            r_label[:, 0] = W - 1 - label[:, 0]
            r_label = r_label.reshape(-1)
            return r_img, r_label
        else:
            return image, label

class CoordResize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        W_ori, H_ori = image.size
        H_target, W_target = self.size[0], self.size[1]
        r_image = TF.resize(image, self.size, self.interpolation)
        label = label.reshape(-1,2)
        r_label = np.zeros_like(label)
        r_label[:, 0] = label[:,0]*W_target / W_ori
        r_label[:, 1] = label[:,1]*H_target / H_ori
        r_label = r_label.reshape(-1)
        return r_image, r_label

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class CoordLabelNormalize:
    def __init__(self):
        pass

    def __call__(self, image, label):
        W,H = image.size
        r_label = label.reshape(-1,2)
        r_label[:,0] /= W
        r_label[:,1] /= H
        r_label = r_label.reshape(-1)
        return image, r_label

def transform_test():
    data_path = './highres_images'
    label_path = './resized_labels'
    save_path = './model'
    is_segment = True

    ##ratio 일정하게 pad하는 custom transform

    transform = tr.Compose([
        #tr.ToPILImage(),
        #tr.RandomPerspective(distortion_scale=0.2, p = 1),
        # tr.RandomRotation(degrees= 15, expand = True),
        # tr.Resize((512,256)),
        CoordCustomPad(512 / 256),
        tr.Resize((512,256)),

        tr.ToTensor(),
        tr.Normalize((0.5,), (0.5,))
    ])

    ##Get DataLoader
    # get label and dataname list
    coords_rel = read_labels(label_path, relative=True)
    data_names = read_data_names(label_path)

    N_all = len(data_names)
    N_val = 0
    N_train = N_all - N_val

    # get train and validation data set
    data_names_train = data_names
    coords_train = coords_rel

    dset_original = SpineDataset(data_path, label_path, data_names_train, coords_rel=coords_train, transform=transform
                                 , is_segment=is_segment)
    loader_original = DataLoader(dataset=dset_original, batch_size=4, shuffle=False)
    #######Displaying the training data

    toPIL = tr.ToPILImage()
    toT = tr.ToTensor()


    for val_data in loader_original:
        imgs = val_data['image'].cpu().to(dtype=torch.float)
        labels = val_data['label'].cpu().to(dtype=torch.float)


        for ind, img in enumerate(imgs):#img is tensor
            plt.figure()
            img = np.asarray(img).reshape(512,256) #go to original image data : nparray (->PILimage)
            t_img = img#transform(img)
            # t_img = toPIL(img)
            # t_img = TF.rotate(t_img, 15)
            # t_img = toT(t_img)

            #img = np.repeat(img.reshape(512,256,1), 3, axis = 2)
            t_img = np.repeat(t_img.reshape(512, 256, 1), 3, axis=2)

            plt.subplot(121)
            plot_image(img)
            plt.title('orig_{}'.format(ind))

            plt.subplot(122)
            plt.imshow(t_img)
            plt.title('trans_{}'.format(ind))
        plt.show()

def rotate_label(label, degree, H, W, new_centerXY=None):

    theta = degree / 180 * np.pi
    s = np.sin(-theta)  # x y 축 바뀜
    c = np.cos(-theta)
    rot_matrix = [[c, s], [-s, c]]  # 회전행렬 transpose

    label = label.reshape(-1, 2)
    origin = np.asarray([W / 2, H / 2])  # x y좌표
    label = label - origin
    label = np.dot(label, rot_matrix)
    if new_centerXY is None:
        new_centerXY = origin
    label += new_centerXY
    label = label.reshape(-1)
    return label

if __name__ == '__main__':

    #def label-to-image
    data_path = './resized_images'
    label_path = './resized_labels'
    labels = read_labels(location = label_path)
    data_names = read_data_names(location = label_path)

    H = 512
    W = 256

    transform = tr.Compose([
        tr.ToTensor()
    ])

    customTransform = [CoordCustomPad(512/256),
                                  #CoordHorizontalFlip(1),
                                  #CoordVerticalFlip(1),
                                  CoordResize((646,210)),
                       CoordLabelNormalize()
                       ]

    ##Get DataLoader
    # get label and dataname list
    N_all = len(data_names)
    # get train and validation data set
    data_names_train = data_names
    coords_train = labels
    dset_original = SpineDataset(data_path, label_path, data_names_train, coords_rel=coords_train, transform=tr.ToTensor()
                                 , is_segment = False)
    loader_original = DataLoader(dataset=dset_original, batch_size=6, shuffle=False)

    randrot = CoordRandomRotate(max_angle = 30, expand = True, is_random = True)

    index = 0
    for val_data in loader_original:
        imgs = val_data['image'].cpu().to(dtype=torch.float)

        for ind, img in enumerate(imgs):#img is tensor
            plt.figure()
            index = index + 1
            img = np.asarray(img).reshape(512,256) #go to original image data : nparray (->PILimage)
            img = tr.ToPILImage()(img)
            label = labels[ind]

            t_img, t_label = (img, label)
            for trans in customTransform:
                t_img, t_label = trans(t_img, t_label)

            #t_label = np.round(t_label, 0)
            # t_img = toPIL(img)
            # t_img = TF.rotate(t_img, 15)
            # t_img = toT(t_img)
            #t_img = t_img.reshape(512,256)
            #img = np.repeat(img.reshape(512,256,1), 3, axis = 2)
            #t_img = np.repeat(t_img.reshape(512, 256, 1), 3, axis=2)

            plt.subplot(121)
            plot_image(img, coord = label)
            plt.title('orig_{}'.format(ind))

            plt.subplot(122)
            #plt.imshow(t_img)
            plot_image(t_img, t_label)
            plt.title('trans_{}'.format(ind))
        plt.show()

    #
    #
    #     dot_image = np.zeros((H, W, 3))
    #     for c,r in label:
    #         rr,cc = circle(r,c,radius = 2)
    #         dot_image[rr, cc, :] = 1
    #     #detecting
    #
    #     blobs_log = blob_log(dot_image)
    #     detected_image = np.zeros((H,W,3))
    #     for ind, blob in enumerate(blobs_log):
    #         y,x,_,_= blob
    #         rr,cc = circle(y,x, radius = 2)
    #         detected_image[rr,cc,:] = 1
    #
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(dot_image)
    #     plt.subplot(122)
    #     plt.imshow(detected_image)
    #     plt.title('all:{}'.format(ind))
    #     plt.show()
    #
    # #make transform
    # #compare result : error? 68점 다 찾는지?