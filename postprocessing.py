import numpy as np
import os
from label_io import read_data_names, read_labels, chw, hwc, write_labels
import matplotlib.pyplot as plt
import pandas as pd
from label_io import to_absolute, to_relative, plot_image
from calc import calc_angle, _get_angle
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

#############################################
#
#   잘못됬음!! label가지고 각도 구할때는 원본이미지 스케일에서 해야 함!
#   테스트 이미지 이터레이터 구현
#   이제 노패드니까 프레딕트도 relative label로 걍 두기
#
#
#   정리는 더 안해도 된다.
###############################################




def calc_poli(x, coeffs):
    res = np.zeros_like(x)
    for coeff in coeffs:
        res *= x
        res += coeff
    return res

def label_fit(label,degree, full = False):
    label = label.reshape(-1,2)
    coeff = np.polyfit(x=label[:, 1], y=label[:, 0], deg=degree)
    out_xaxis = calc_poli(label[:,1], coeff)
    out_label = np.concatenate((out_xaxis.reshape(-1,1), label[:,1].reshape(-1,1)), axis = 1)
    out_label = out_label.reshape(-1)
    if full:
        return out_label, coeff
    else:
        return out_label

def derivative_poli(coeff):
    if len(coeff) ==1:
        return [0]
    else:
        N = len(coeff)
        der = []
        degree = N-1
        for ind in range(N-1):
            dc = degree * coeff[ind]
            der.append(dc)
            degree -=1
    return der

def post_way1(pred_path, loader_test):
    #pred_path = './model/RH_SM_all_ep1999'
    preds = label_sort(read_labels(location = pred_path))

    #all processing is relative
    #preds = to_relative(preds)

    fit_degree = 6

    ind = -1
    processing_log = []
    for val_data in loader_test:
        imgs = val_data['image'].cpu().numpy()  # for batch size 1
        gts = val_data['label'].cpu().numpy()
        for i, img in enumerate(imgs):
            ind += 1
            gt = gts[i]
            H, W = 512, 256

            pred = preds[ind]
            pred = pred.reshape(34, 2, 2)

            ########################################################
            #
            #   Left fit과 right fit 만들고 왼쪽 오른쪽 각각에 대해
            #   landmark 구하기
            #   해당 랜드마크 가지고 angle prediction
            #
            #
            #
            #
            ########################################################

            left = pred[:, 0, :].reshape(-1, 2)
            left_fit, coeff_l = label_fit(left, fit_degree, full=True)

            right = pred[:, 1, :].reshape(-1, 2)
            right_fit, coeff_r = label_fit(right, fit_degree, full=True)

            fitted_pred = np.concatenate((left_fit.reshape(-1, 1, 2), right_fit.reshape(-1, 1, 2))
                                         , axis=1)
            fitted_pred = fitted_pred.flatten()
            pred = pred.flatten()

            angles_g, pos_g, _, mid_lines_g, _ = calc_angle(gt, (1, 1))
            angles_p, pos_p, _, mid_lines_p, _ = calc_angle(fitted_pred, (1, 1))

            angles_g = angles_g[0]
            angles_p = angles_p[0]

            angle_err = np.abs(angles_g - angles_p) / angles_g

            pos_g = pos_g[0:2]
            pos_p = pos_p[0:2]

            # right = np.tile(lab[:,1,:].expand_dims(1), (1,2,1))
            C, H, W = img.shape
            plt.figure()

            for pos in pos_g:
                plt.plot(mid_lines_g[2 * pos:2 * pos + 2, 0] * W, mid_lines_g[2 * pos:2 * pos + 2, 1] * H, 'g')
            for pos in pos_p:
                plt.plot(mid_lines_p[2 * pos:2 * pos + 2, 0] * W, mid_lines_p[2 * pos:2 * pos + 2, 1] * H, 'r')

            # fit line 추가
            d = 1 / 512
            pred = pred.reshape(34, 2, 2)
            minyl = pred[0, 0, 1]
            minyr = pred[0, 1, 1]
            maxyl = pred[33, 0, 1]
            maxyr = pred[33, 1, 1]
            left_r = np.arange(minyl, maxyl, d)
            right_r = np.arange(minyr, maxyr, d)
            left_c = calc_poli(left_r, coeff_l)
            right_c = calc_poli(right_r, coeff_r)
            left_line = np.concatenate((left_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)
            right_line = np.concatenate((right_c.reshape(-1, 1), right_r.reshape(-1, 1)), axis=1)

            plt.plot(left_line[:, 0] * W, left_line[:, 1] * H, color='magenta', alpha=0.5)
            plt.plot(right_line[:, 0] * W, right_line[:, 1] * H, color='magenta', alpha=0.5)

            plot_image(img, coord_red=fitted_pred, coord_gr=gt, coord_cy=pred)

            # fit line 윤곽추가 맨윗좌표~맨아랫좌표
            title = 'GT%.1f %d %d PR%.1f %d %d ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
                                                             angles_p, pos_p[0], pos_p[1], angle_err * 100)
            plt.title('{}_R:PRED, G:GT'.format(ind) + '\n' + title)
            # plt.show()
            save_name = 'way1_{}'.format(ind)
            plt.savefig(os.path.join(pred_path, save_name + '.jpg'))
            plt.close()

            processing_log.append(dict(
                pos_GT0=pos_g[0], pos_GT1=pos_g[1], pos_PR0=pos_p[0], pos_PR1=pos_p[1],
                angles_g=angles_g, angles_p=angles_p, err=angle_err
            ))

            ########################################################
            #
            #   Left fit과 right fit의 중점을 midpoint curve라고 생각하기
            #
            ########################################################
    df = pd.DataFrame(processing_log)
    df.to_csv(os.path.join(pred_path, save_name + '.csv'))

    ##############################
    #  WAY1
    #
    #
    #
    #########################
def post_way2(pred_path, loader_test):
    #pred_path = './model/RH_SM_all_ep1999'
    #data_path = './_images'
    #label_path = './highres_labels'
    preds = label_sort(read_labels(label_location=pred_path, title = 'labels_pred'))
    #preds = to_relative(preds)
    #write_labels(sorted_preds, location=pred_path, title='labels_sorted')

    #상대좌표로 모든걸 프로세싱
    fit_degree = 6

    processing_log = []
    ind = -1
    for val_data in loader_test:
        imgs = val_data['image'].cpu().numpy()  # for batch size 1
        gts = val_data['label'].cpu().numpy()
        for i, img in enumerate(imgs):
            ind +=1
            gt = gts[i]
            H, W = 512, 256

            pred = preds[ind]
            pred = pred.reshape(34, 2, 2)

            ########################################################
            #
            #   Left fit과 right fit의 중점을 midpoint curve라고 생각하기
            #
            ########################################################
            left = pred[:, 0, :].reshape(-1, 2)
            left_fit, _coeff_l = label_fit(left, fit_degree, full=True)

            right = pred[:, 1, :].reshape(-1, 2)
            right_fit, _coeff_r = label_fit(right, fit_degree, full=True)

            coeff = (_coeff_l + _coeff_r) / 2
            der_coeff = derivative_poli(coeff)
            der_coeff = [c / (H / W) for c in der_coeff]  # 상대좌표는 해줘야함

            pred = pred.reshape(17, 4, 2)
            midpoints = np.zeros((17, 2))

            # midpoint 추정을 피팅한걸로 해서 할 수도 있다
            ############################################
            #
            #   midpoint curve의 도함수를 구하고
            #   추정랜드마크의 r값을 통해 해당 기울기 구함
            #   그다음 그거가지고 angle구함
            #
            #############################################
            for i, pnts in enumerate(pred):
                midpoints[i] = np.average(pnts, axis=0)
            midpoints_x = midpoints[:, 1]
            der_y = calc_poli(midpoints_x, der_coeff)
            atans = np.arctan(der_y) * 180 / np.pi
            # 1,-m이 slope vector
            slopes = np.ones((17, 2))
            slopes[:, 1] = -der_y
            angles = _get_angle(slopes, slopes)
            pos_p = np.argmax(angles)
            pos_p = np.unravel_index(pos_p, angles.shape)
            angles_p = angles[pos_p] / np.pi * 180

            angles_g, pos_g, _, mid_lines_g, _ = calc_angle(gt, (1, 1))
            angles_g = angles_g[0]

            angle_err = np.abs(angles_g - angles_p) / angles_g

            pos_g = pos_g[0:2]
            pos_p = pos_p[0:2]

            # right = np.tile(lab[:,1,:].expand_dims(1), (1,2,1))
            C, H, W = img.shape
            plt.figure()

            for pos in pos_g:
                plt.plot(mid_lines_g[2 * pos:2 * pos + 2, 0] * W, mid_lines_g[2 * pos:2 * pos + 2, 1] * H, 'g')

            for pos in pos_p:
                # plt.plot(mid_lines_p[2 * pos:2 * pos + 2, 0] * W, mid_lines_p[2 * pos:2 * pos + 2, 1] * H, 'r')
                plt.plot([midpoints[pos, 0] * W], [midpoints[pos, 1] * H], 'ro', markersize=3.5)

            #################################
            # fit line 추가
            #
            #################################
            d = 1 / 512
            pred = pred.reshape(34, 2, 2)
            minyl = pred[0, 0, 1]
            minyr = pred[0, 1, 1]
            maxyl = pred[33, 0, 1]
            maxyr = pred[33, 1, 1]

            left_r = np.arange(minyl, maxyl, d)
            right_r = np.arange(minyr, maxyr, d)
            left_c = calc_poli(left_r, _coeff_l)
            right_c = calc_poli(right_r, _coeff_r)
            left_line = np.concatenate((left_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)
            right_line = np.concatenate((right_c.reshape(-1, 1), right_r.reshape(-1, 1)), axis=1)

            mid_c = calc_poli(left_r, coeff)
            mid_line = np.concatenate((mid_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)

            # plt.plot(left_line[:, 0] * W, left_line[:, 1] * H, color='magenta', alpha=0.5)
            # plt.plot(right_line[:, 0] * W, right_line[:, 1] * H, color='magenta', alpha=0.5)

            plot_image(img, coord_red=midpoints.reshape(-1), coord_gr=gt, coord_cy=pred, line_red=mid_line)

            # fit line 윤곽추가 맨윗좌표~맨아랫좌표
            # title = 'GT%.1f %d %d PR%.1f %d %d ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
            #                                                  angles_p, pos_p[0], pos_p[1], angle_err * 100)
            title = 'GT%.1f %d %d PR%.1f %.1f %.1f ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
                                                                 angles_p, atans[pos_p[0]], atans[pos_p[1]],
                                                                 angle_err * 100)
            plt.title('{}_R:PRED, G:GT'.format(ind) + '\n' + title)
            # plt.show()
            save_name = 'way2_{}'.format(ind)
            plt.savefig(os.path.join(pred_path, save_name + '.jpg'))
            plt.close()

            processing_log.append(dict(
                pos_GT0=pos_g[0], pos_GT1=pos_g[1], pos_PR0=pos_p[0], pos_PR1=pos_p[1],
                angles_g=angles_g, angles_p=angles_p, err=angle_err
            ))

    df = pd.DataFrame(processing_log)
    df.to_csv(os.path.join(pred_path, save_name + '.csv'))



