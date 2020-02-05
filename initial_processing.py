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

from label_io import write_labels, write_data_names, read_labels, read_data_names, plot_image, read_images, label_sort


##############################
#
#   Preprocessing data for segmentation
#
#   손대지말자 어차피 쓸데없음!
#
#   혼동 X:
#   location은 *.csv 빼고, path는 전체 path, dest는 저장장소!!
#
##############################

def preprocessing():

    #여기서 데이터 프리프로세싱 작업 하기

    #path declaration
    # image_dest_resized = './resized_images'
    # label_dest_resized = './resized_labels'

    image_path = './highres_images'
    label_path = './highres_labels'

    data_names = read_data_names(label_location= label_path)
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

def prepare_label():
    # read matlab label and save to csv
    # image_location = './train_images'
    # label_location_mat = './train_labels_mat'
    # label_save_dest = './train_labels_mat'

    image_location = './test_images'
    label_location_mat = './Old/boostnet_labeldata/labels/test'
    label_save_dest = './test_labels_mat'

    label_path_all = os.listdir(label_location_mat)

    label_names = []
    output_list = []

    for label_name in label_path_all:
        if '.mat' == label_name[-4:]:
            label_names.append(label_name)

    data_names = [label_name[:-4] for label_name in label_names]

    labels = []
    for data_no, data_name in enumerate(data_names):
        image = Image.open(os.path.join(image_location, data_name))
        lb = spio.loadmat(os.path.join(label_location_mat, label_names[data_no]))
        coord = np.array(lb['p2'], dtype=np.float).reshape(-1)
        np_image = np.array(image)
        H, W = np_image.shape
        labels.append(coord.flatten())

    labels = np.array(labels)
    labels = label_sort(labels)
    write_labels(labels, label_location=label_save_dest)
    write_data_names(data_names, label_location=label_save_dest)

def draw_seg(coord, H, W):
    seg_image = np.zeros((H, W, 3))
    coord_rev = coord.reshape(-1, 2, 2)
    pol = np.concatenate((coord_rev[0, 0, :].reshape(1, 2), coord_rev[:, 1, :], coord_rev[::-1, 0, :]), axis=0)
    rr, cc = polygon(pol[:, 1], pol[:, 0], seg_image.shape)
    seg_image[rr, cc, :] = 1
    seg_image = seg_image[:, :, 0]
    return seg_image

if __name__ == '__main__':
    label_test_frommat = read_labels('./test_labels_mat')
    label_test_my = read_labels('./test_labels')

    # label_test_frommat = read_labels('./test_labels', title = 'landmarks')
    # label_test_frommat = label_test_frommat.reshape(-1,2,68).transpose(0,2,1).reshape(-1,136)

    equal = label_test_frommat == label_test_my

    diff = label_test_my.size - np.sum(equal)
    print('unsorted label differs in {}'.format(diff))

    sorted_label = label_sort(label_test_frommat)
    equal = sorted_label == label_test_my
    diff = label_test_my.size - np.sum(equal)

    print('sorted label differs in {}'.format(diff))

    # ####################
    # ####
    # ####    Check label order
    #
    #
    #
    # test_data_location = './test_images'
    # test_label_location = './test_labels'
    # test_labels = read_labels(test_label_location)
    # test_data_names = read_data_names(test_label_location)
    # test_images = read_images(test_data_location, test_data_names)
    #
    # train_data_location = './train_images'
    # train_label_location = './train_labels'
    #
    # #train_labels = read_labels(train_label_location)
    # train_labels = read_labels(train_label_location, title = 'labels_m')
    # train_data_names = read_data_names(train_label_location)
    # train_images = read_images(train_data_location, train_data_names)
    #
    # # train_labels_nosort = read_labels(train_label_location, title = 'labels_nosort')
    # # train_labels[1] = train_labels_nosort[1]
    # # train_labels[75] = train_labels_nosort[75]
    # #
    # # write_labels(train_labels, label_location= train_label_location, title = 'labels_finver')
    #
    #
    # error_datas = []
    # #label 유효성 검사
    # for ind, image in enumerate(test_images):
    #     gt = test_labels[ind]
    #
    #     ##label 유효성 검사
    #     gt = gt.reshape(34,2,2)
    #     mask = np.zeros_like(gt)
    #
    #     is_error = False
    #     #left x < right x
    #     for i in range(34):
    #         if not gt[i,0,0] <= gt[i,1,0]:
    #             is_error = True
    #             mask[i,0,:] = 1
    #             mask[i,1,:] = 1
    #             print('{} : left right '.format(ind))
    #
    #     #top y < bot y (matrix coordinate)
    #     for i in range(33):
    #         if not gt[i,0,1] <= gt[i+1,0,1]:
    #             mask[i,0,:] = 1
    #             mask[i+1,0,:] = 1
    #             is_error = True
    #             print('{} : top bot for left '.format(ind))
    #         if not gt[i,1,1] <= gt[i+1,1,1]:
    #             mask[i, 1, :] = 1
    #             mask[i + 1, 1, :] = 1
    #             is_error = True
    #             print('{} : top bot for right '.format(ind))
    #
    #
    #
    #     if is_error:
    #         notmask = np.ones_like(gt) - mask
    #         gt_m = gt * notmask + np.ones_like(gt)
    #         gt_error = gt*mask + np.ones_like(gt)
    #
    #         plt.figure()
    #         plt.title('ind : {}'.format(ind))
    #
    #         plot_image(image, coord_red=gt_m.flatten(), coord_gr = gt_error.flatten())
    #         plt.show()
    #     # plt.figure()
    #     # plot_image(image, coord_red=gt[:,0,:].flatten(), coord_gr= gt[:,1,:].flatten())
    #     # plt.show()