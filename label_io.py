import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
from PIL import Image

#############################
#
#       리팩토링 완료
#
##############################

def plot_image(image, coord_red = None, coord_gr = None, coord_bl = None,
               line_red = None, line_gr = None, line_bl = None,
               segmap = None, ref_segmap = None, alpha = 0.3, off_scaling = False):
    #각각 coord들을 독립적으로 변경

    if not type(image) == type(np.ones(2)): #for PIL image
        image = np.array(image)
    if np.max(image) >3.0:#scale 255일경우
        image = image/255
    if len(image.shape) ==2: #for HW
        image = hwc(image)
    if image.shape[0] ==1: #for CHW
        _,H,W = image.shape
        image = hwc(image.reshape(H,W))

    if not off_scaling:
        image = denormalize_image(image)

    H,W,C = image.shape

    green = [[0.1,1,0.1,1]]

    #imshow는 scatter 뒤에 해야 함!

    if coord_red is not None:
        coord_red = np.copy(coord_red.reshape(-1, 2))
        if coord_red.flatten()[0] < 1:
            coord_red = to_absolute(coord_red)
        plt.scatter(coord_red[:, 0], coord_red[:, 1], s=1.2, c='red')

    if coord_gr is not None:
        coord_gr = np.copy(coord_gr.reshape(-1, 2))
        if coord_gr.flatten()[0] < 1:
            coord_gr = to_absolute(coord_gr)
        plt.scatter(coord_gr[:, 0], coord_gr[:, 1], s=1.2, c=green)

    if coord_bl is not None:
        coord_bl = np.copy(coord_bl.reshape(-1, 2))
        if coord_bl.flatten()[0] < 1:
            coord_bl = to_absolute(coord_bl)
        plt.scatter(coord_bl[:, 0], coord_bl[:, 1], s=1.2, c='blue')

    if line_red is not None:
        line = np.copy(line_red.reshape(-1, 2))
        if line.flatten()[0] <1:
            line = to_absolute(line)
        plt.plot(line[:, 0], line[:, 1], color ='red', alpha = 0.5)

    if line_gr is not None:
        line = np.copy(line_gr.reshape(-1, 2))
        if line.flatten()[0] <1:
            line = to_absolute(line)
        plt.plot(line[:, 0], line[:, 1], color =green, alpha = 0.5)

    if line_bl is not None:
        line = np.copy(line_bl.reshape(-1, 2))
        if line.flatten()[0] <1:
            line = to_absolute(line)
        plt.plot(line[:, 0], line[:, 1], color ='blue', alpha = 0.5)

    plt.imshow(image)

    if segmap is None and coord_red is None:
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

    plt.subplots_adjust(left=0.00, bottom=0.00, right=1 - 0.00, top=1 - 0.10, hspace=0, wspace=0)
    plt.axis('off')

def label_sort_x(labels):
    labels = labels.reshape(-1,34,2,2)
    for label in labels:
        for i in range(34):
            #   if not left coord x < right coord x
            if not label[i][0][0] <= label[i][1][0]:
                tmp = label[i][1].copy()
                label[i][1] = label[i][0].copy()
                label[i][0] = tmp
    return labels.reshape(-1,136)

def label_sort_y(labels):
    #1st axis : 증가하는 axis가 0임 즉 column별로
    labels_rev = np.copy(labels.reshape(-1,34,2,2))

    label_list = []
    #y sort
    for label in labels_rev:
        left = label[:,0,:]
        left = [c for c in left]
        right = label[:,1,:]
        right = [c for c in right]
        left = sorted(left, key = lambda x:x[1])
        right = sorted(right, key = lambda x:x[1])
        left = np.array(left)
        right = np.array(right)
        label_list.append(np.concatenate([left, right], axis = 1))
    labels_rev = np.array(label_list)
    labels_rev = labels_rev.reshape(-1,136)
    return labels_rev

def label_sort(labels):
    labels = label_sort_x(labels)
    labels = label_sort_y(labels)
    labels = label_sort_x(labels)
    return labels

def to_relative(label, H = 512, W = 256):
    #labels인지 label인지
    all_size = label.size
    shape = label.shape

    if np.average(label) <1:
        warnings.warn('the label is already relative with indice {} '.format(label.flatten()[0]))

    rep = int(all_size/2)
    factor = np.tile(np.asarray([W,H]), (rep,))
    rel_label = np.copy(label)
    rel_label = rel_label.flatten()/factor
    rel_label = rel_label.reshape(shape)
    return rel_label

def to_absolute(label, H = 512, W = 256):
    # labels인지 label인지
    all_size = label.size
    shape = label.shape

    if np.average(label) > 3:
        warnings.warn('the label is already absolute with indice {} '.format(label.flatten()[0]))

    rep = int(all_size / 2)
    factor = np.tile(np.asarray([W, H]), (rep,))
    abs_label = np.copy(label)
    abs_label = abs_label.flatten() * factor
    abs_label = abs_label.reshape(shape).astype(int)
    return abs_label

def read_images(data_location, data_names):
    image_list = []
    for data_name in data_names:
        im = Image.open(os.path.join(data_location, data_name))
        im = hwc(np.asarray(im))
        image_list.append(im)
    return image_list

def read_data_names(label_location):
    path = os.path.join(label_location, 'data_names.csv')
    df = pd.read_csv(path, index_col=False, header=None)
    data_names = [df.iloc[i][0] for i in range(len(df))]
    return data_names

def write_data_names(data_names, label_location):
    path = os.path.join(label_location, 'data_names.csv')
    pd.DataFrame(data_names).to_csv(path, header=False, index=False)

def read_labels(label_location, title = None):
    if title is None:
        title = 'labels'

    if title[-4:] != '.csv':
        title = title + '.csv'

    path = os.path.join(label_location, title)
    labels = pd.read_csv(path, header=None, index_col=False)
    return np.asarray(labels)

def write_labels(labels, label_location, relative = False, title = None):
    if title is None:
        if relative == False:
            title = 'labels'
        else:
            raise Exception('Do not save and load relative label')

    if title[-4:] != '.csv':
        title = title + '.csv'

    path = os.path.join(label_location, title)
    pd.DataFrame(labels).to_csv(path, header=False, index=False)

def hwc(nparr):
    H,W = nparr.shape
    img = nparr.reshape((H,W,1)).repeat(3, axis = 2)
    return img
def chw(nparr):
    H,W = nparr.shape
    img = nparr.reshape((1,H,W)).repeat(3, axis = 0)
    return img

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

#def write_relative_label():
#     #read the absolute label and write relative label
#     label_path = './resized_labels'
#     data_location = './resized_images'
#     data_names = read_data_names(location=label_path)
#     labels_abs = read_labels(location= label_path)
#
#     label_list = []
#     for ind, label in enumerate(labels_abs):
#         img = Image.open(os.path.join(data_location, data_names[ind]))
#         H,W = np.asarray(img).shape
#         #
#         #
#
#         label = label.reshape(-1, 2)
#         label[:, 0] /= W
#         label[:, 1] /= H
#         label = label.reshape(-1)
#         label_list.append(label)
#     label_rel = np.asarray(label_list)
#     write_labels(label_rel, location= label_path, relative= True)

#if __name__  == '__main__':
# preprocessing()
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