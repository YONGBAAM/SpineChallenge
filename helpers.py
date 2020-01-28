import matplotlib.pyplot as plt

from skimage.draw import line, polygon, circle, circle_perimeter, ellipse
from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
import scipy.io as spio
import pandas as pd

def draw_seg(coord, H, W):
    seg_image = np.zeros((H, W, 3))
    coord_rev = coord.reshape(-1, 2, 2)
    pol = np.concatenate((coord_rev[0, 0, :].reshape(1, 2), coord_rev[:, 1, :], coord_rev[::-1, 0, :]), axis=0)
    rr, cc = polygon(pol[:, 1], pol[:, 0], seg_image.shape)
    seg_image[rr, cc, :] = 1
    seg_image = seg_image[:,:,0]
    return seg_image

def read_data_names(location):
    path = os.path.join(location, 'data_names.csv')
    df = pd.read_csv(path, index_col=False, header=None)
    data_names = [df.iloc[i][0] for i in range(len(df))]
    return data_names

def write_data_names(data_names, location):
    path = os.path.join(location, 'data_names.csv')
    pd.DataFrame(data_names).to_csv(path, header=False, index=False)

def read_labels(location, relative = False):
    if relative == False:
        path = os.path.join(location, 'labels.csv')
    else:
        path = os.path.join(location, 'labels_rel.csv')
    labels = pd.read_csv(path, header=None, index_col=False)
    return np.asarray(labels)

def write_labels(labels, location, relative = False):
    if relative == False:
        path = os.path.join(location, 'labels.csv')
    else:
        path = os.path.join(location, 'labels_rel.csv')
    pd.DataFrame(labels).to_csv(path, header=False, index=False)

def hwc(nparr):
    H,W = nparr.shape
    img = nparr.reshape((H,W,1)).repeat(3, axis = 2)
    return img

def chw(nparr):
    H,W = nparr.shape
    img = nparr.reshape((1,H,W)).repeat(3, axis = 0)
    return img

##############################
#
#   Preprocessing data for segmentation
#
#   혼동 X:
#   location은 *.csv 빼고, path는 전체 path, dest는 저장장소!!
#
##############################

def plot_image(image, coord = None, segmap = None, ref_coord = None, ref_segmap = None, alpha = 0.3, off_scaling = False):
    #다른기능은 생략하기로 하고
    #scale 255일경우
    if not type(image) == type(np.ones(2)): #for PIL image
        image = np.array(image)
    if np.max(image) >3.0:
        image = image/255
    if len(image.shape) ==2: #for HW
        image = hwc(image)
    if image.shape[0] ==1: #for CHW
        _,H,W = image.shape
        image = hwc(image.reshape(H,W))

    if not off_scaling:
        image = denormalize_image(image)

    H,W,C = image.shape

    #imshow는 scatter 뒤에 해야 함!

    if coord is not None:
        coord = np.copy(coord.reshape(-1, 2))
        if coord[0][1] <1:
            #Coord는 상대좌표
            coord[:,0] *= W
            coord[:,1] *= H
        plt.scatter(coord[:, 0], coord[:, 1], s=1.2, c='red')
        if ref_coord is not None:
            ref_coord = np.copy(ref_coord.reshape(-1, 2))
            if ref_coord[0][1] < 1:
                # Coord는 상대좌표
                ref_coord[:, 0] *= W
                ref_coord[:, 1] *= H
            plt.scatter(ref_coord[:, 0], ref_coord[:, 1], s=1.2, c='green')
        plt.imshow(image)

    if segmap is None and coord is None:
        plt.imshow(image)

    if segmap is not None:
        segmap = segmap.reshape(H, W)
        segmap = hwc(segmap)
        segmap[:, :, (1, 2)] = 0

        if ref_segmap is not None:
            ref_segmap = ref_segmap.reshape(H, W)
            ref_segmap = hwc(ref_segmap)
            ref_segmap[:, :, (0, 2)] = 0  # green for gt
            plt.imshow(image * (1 - 2 * alpha) + segmap * alpha + ref_segmap * alpha)
        else:
            plt.imshow(image * (1 - alpha) + segmap * alpha)




def preprocessing():
    #여기서 데이터 프리프로세싱 작업 하기

    #path declaration
    # image_dest_resized = './resized_images'
    # label_dest_resized = './resized_labels'

    image_path = './highres_images'
    label_path = './highres_labels'

    data_names = read_data_names(location = label_path)
    labels = read_labels(label_path)

    #testing image and labels
    # plt.figure()
    # for i,ind in enumerate([35,67,25,235]):
    #     plt.subplot(221+i)
    #     image = Image.open(os.path.join(image_path, data_names[ind]))
    #     segmap = np.load(os.path.join(label_path, data_names[ind] + '.npy'))
    #     plot_image(image, coord = labels[ind], segmap=segmap )
    # plt.show()

    # #pad image
    # resized_H = 512
    # desired_W = 256

    label_list = []

    for data_no, data_name in enumerate(data_names):
        image = Image.open(os.path.join(image_path, data_name))
        np_image = np.asarray(image)
        label = labels[data_no]

        H, W = np_image.shape
        resized_H = H
        desired_W = W
        # resize_ratio = resized_H / H
        # left_pad = int((desired_W - int(W * resize_ratio)) / 2)
        # right_pad = desired_W - int(W * resize_ratio) - left_pad
        #
        # label_rev = label.reshape(-1,2)
        # label_rev *= resize_ratio
        # label_rev[:,0] += left_pad
        # label_rev = label_rev.reshape(-1)
        # label_list.append(label_rev)
        #
        # im_resize = image.resize((int(W * resize_ratio), int(resized_H)))
        # im_pad = ImageOps.expand(im_resize, (left_pad, 0, right_pad, 0))
        # im_pad.save(os.path.join(image_dest_resized, data_name))

        segmap = draw_seg(label, resized_H, desired_W)
        np.save(os.path.join(label_path, data_name + '.npy'), segmap)

        if data_no%100 ==0:
            print(np.asarray(image).shape)
            plt.figure()
            plot_image(image, label, segmap=segmap)
    plt.show()
    # label_rev_all = np.asarray(label_list)
    # write_labels(label_rev_all, location = label_dest_resized)
    # write_data_names(data_names, location = label_dest_resized)

def write_relative_label():
    #read the absolute label and write relative label
    label_path = './resized_labels'
    data_location = './resized_images'
    data_names = read_data_names(location=label_path)
    labels_abs = read_labels(location= label_path)

    label_list = []
    for ind, label in enumerate(labels_abs):
        img = Image.open(os.path.join(data_location, data_names[ind]))
        H,W = np.asarray(img).shape
        #
        #

        label = label.reshape(-1, 2)
        label[:, 0] /= W
        label[:, 1] /= H
        label = label.reshape(-1)
        label_list.append(label)
    label_rel = np.asarray(label_list)
    write_labels(label_rel, location= label_path, relative= True)

def denormalize_image(image):
    max = np.argmax(image)
    max = np.unravel_index(max, image.shape)

    min = np.argmin(image)
    min = np.unravel_index(min, image.shape)

    minval = image[min]
    maxval = image[max]

    offset = -minval
    scale = 1/(maxval - minval)

    return (image + offset) * scale


if __name__  == '__main__':
    preprocessing()
    # plt.figure()
    # img = Image.open('./resized_images/sunhl-1th-02-Jan-2017-162 A AP.jpg')
    # img = np.asarray(img)/255.0
    # img = img/2 + 0.1
    #
    # plt.imshow(img)
    # label = read_labels(location='./resized_labels', relative=True)
    # label = label[0].reshape(-1, 2)
    # label[:, 0] *= 256
    # label[:, 1] *= 512
    # plt.scatter(label[:, 0], label[:, 1])
    # plt.show()
    # plt.figure()
    # plot_image(img, coord = label)
    # plt.show()