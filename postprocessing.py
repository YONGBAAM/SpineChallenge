import numpy as np
import os
from label_io import read_data_names, read_labels, chw, hwc, write_labels
import matplotlib.pyplot as plt
import pandas as pd
from label_io import to_absolute, to_relative, plot_image, label_sort, read_images
from calc import calc_angle_old, _get_angle, calc_angle, get_mid_points, get_vec_lines
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import time

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
        res = res + coeff
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

#with predicted coordinate
#integrating
def post_way1(label_pred_abs, degree = 6, full = False):
    #pred_path = './model/RH_SM_all_ep1999'

    ####    Get ablsolute prediction
    fit_degree = degree
    pred = label_pred_abs
    pred = pred.reshape(34, 2, 2)

    ########################################################
    #   Left fit과 right fit 만들고 왼쪽 오른쪽 각각에 대해
    #   landmark 구하기
    #   해당 랜드마크 가지고 angle prediction
    ########################################################

    left = pred[:, 0, :]
    left_fit, coeff_l = label_fit(left, fit_degree, full=True)

    right = pred[:, 1, :]
    right_fit, coeff_r = label_fit(right, fit_degree, full=True)

    fitted_preds = np.concatenate((left_fit.reshape(-1, 1, 2), right_fit.reshape(-1, 1, 2))
                                  , axis=1)
    fitted_preds = fitted_preds.flatten().astype(int)

    #####################
    #   여기부턴 밖에서 하기
    # angles_g, pos_g, _, mid_lines_g, _ = calc_angle(gt, image_size =(H,W))
    # angles_p, pos_p, _, mid_lines_p, _ = calc_angle(fitted_pred, (H, W))
    #
    # angles_g = angles_g[0]
    # angles_p = angles_p[0]
    #
    # angle_err = np.abs(angles_g - angles_p) / angles_g
    #
    # pos_g = pos_g[0:2]
    # pos_p = pos_p[0:2]
    #
    # # right = np.tile(lab[:,1,:].expand_dims(1), (1,2,1))
    # plt.figure()
    #
    # for pos in pos_g:
    #     plt.plot(mid_lines_g[2 * pos:2 * pos + 2, 0] * W, mid_lines_g[2 * pos:2 * pos + 2, 1] * H, 'g')
    # for pos in pos_p:
    #     plt.plot(mid_lines_p[2 * pos:2 * pos + 2, 0] * W, mid_lines_p[2 * pos:2 * pos + 2, 1] * H, 'r')

    # fit line 추가
    d = 1
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


    params = dict(
        left_coeff = coeff_l, right_coeff = coeff_r,
        left_line = left_line, right_line = right_line

    )
    if full:
        return fitted_preds, params
    else:
        return fitted_preds

        # plt.plot(left_line[:, 0], left_line[:, 1], color='magenta', alpha=0.3)
        # plt.plot(right_line[:, 0], right_line[:, 1], color='magenta', alpha=0.3)

        # plot_image(img, coord_red=fitted_pred, coord_gr=gt)

        # fit line 윤곽추가 맨윗좌표~맨아랫좌표
    #     title = 'GT%.1f %d %d PR%.1f %d %d ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
    #     #                                                      angles_p, pos_p[0], pos_p[1], angle_err * 100)
    #     #     plt.title('{}_R:PRED, G:GT'.format(ind) + '\n' + title)
    #     #     #plt.show()
    #     #     save_name = 'way1_{}'.format(ind)
    #     #     plt.savefig(os.path.join(pred_path, save_name + '.jpg'))
    #     #     plt.close()
    #     #
    #     #     processing_log.append(dict(
    #     #         pos_GT0=pos_g[0], pos_GT1=pos_g[1], pos_PR0=pos_p[0], pos_PR1=pos_p[1],
    #     #         angles_g=angles_g, angles_p=angles_p, err=angle_err
    #     #     ))
    #     #
    #     #     ########################################################
    #     #     #
    #     #     #   Left fit과 right fit의 중점을 midpoint curve라고 생각하기
    #     #     #
    #     #     ########################################################
    #     # df = pd.DataFrame(processing_log)
    #     # df.to_csv(os.path.join(pred_path, save_name + '.csv'))

    ##############################
    #  WAY1
    #
    #
    #
    #########################
def post_way2(label_pred_abs, degree = 6, full = False):

    # pred_path = './model/RH_SM_all_ep1999'

    ####    Get ablsolute prediction
    fit_degree = degree
    pred = label_pred_abs

    pred = pred.reshape(34, 2, 2)

    left = pred[:, 0, :]
    _, coeff_l = label_fit(left, fit_degree, full=True)
    right = pred[:, 1, :]
    _, coeff_r = label_fit(right, fit_degree, full=True)

    coeff = list((coeff_l + coeff_r) / 2)
    der_coeff = derivative_poli(coeff)
    # 절대좌표니까 괜찮음상대좌표는 해줘야함

    pred = pred.reshape(17, 4, 2)
    midpoints4 = np.zeros((17, 2))

    # midpoint 추정을 피팅한걸로 해서 할 수도 있다
    ############################################
    #
    #   midpoint curve의 도함수를 구하고
    #   추정랜드마크의 r값을 통해 해당 기울기 구함
    #   그다음 그거가지고 angle구함
    #
    #############################################
    for i, pnts in enumerate(pred):
        midpoints4[i] = np.average(pnts, axis=0)
    midpoints_x = midpoints4[:, 1]

    der_y = calc_poli(midpoints_x, der_coeff)
    # 1,-m이 slope vector
    slopes = np.ones((17, 2))
    slopes[:, 1] = -der_y

    # for pos in pos_g:
    #     plt.plot(mid_lines_g[2 * pos:2 * pos + 2, 0] * W, mid_lines_g[2 * pos:2 * pos + 2, 1] * H, 'g')
    #
    # for pos in pos_p:
    #     # plt.plot(mid_lines_p[2 * pos:2 * pos + 2, 0] * W, mid_lines_p[2 * pos:2 * pos + 2, 1] * H, 'r')
    #     plt.plot([midpoints4[pos, 0] * W], [midpoints4[pos, 1] * H], 'ro', markersize=3.5)

    ####    Ploting
    d = 1
    pred = pred.reshape(34, 2, 2)
    minyl = pred[0, 0, 1]
    minyr = pred[0, 1, 1]
    miny = int((minyl + minyr) / 2)
    maxyl = pred[33, 0, 1]
    maxyr = pred[33, 1, 1]
    maxy = int((maxyl + maxyr) / 2)
    middle_r = np.arange(miny, maxy, d)
    middle_c = calc_poli(middle_r, coeff)
    middle_line = np.concatenate((middle_c.reshape(-1, 1), middle_r.reshape(-1, 1)), axis=1)

    params = dict(
        coeff = coeff, line = middle_line, middle_vec = midpoints4
    )
    if full:
        return slopes, params
    else:
        return slopes
    #상대좌표로 모든걸 프로세싱
    # fit_degree = 6
    #
    # processing_log = []
    # ind = -1
    # for val_data in loader_test:
    #     imgs = val_data['image'].cpu().numpy()  # for batch size 1
    #     gts = val_data['label'].cpu().numpy()
    #     for i, img in enumerate(imgs):
    #         ind +=1
    #         gt = gts[i]
    #         H, W = 512, 256
    #
    #         pred = preds[ind]
    #         pred = pred.reshape(34, 2, 2)
    #
    #         ########################################################
    #         #
    #         #   Left fit과 right fit의 중점을 midpoint curve라고 생각하기
    #         #
    #         ########################################################
    #         left = pred[:, 0, :].reshape(-1, 2)
    #         left_fit, _coeff_l = label_fit(left, fit_degree, full=True)
    #
    #         right = pred[:, 1, :].reshape(-1, 2)
    #         right_fit, _coeff_r = label_fit(right, fit_degree, full=True)
    #
    #         coeff = (_coeff_l + _coeff_r) / 2
    #         der_coeff = derivative_poli(coeff)
    #         der_coeff = [c / (H / W) for c in der_coeff]  # 상대좌표는 해줘야함
    #
    #         pred = pred.reshape(17, 4, 2)
    #         midpoints4 = np.zeros((17, 2))
    #
    #         # midpoint 추정을 피팅한걸로 해서 할 수도 있다
    #         ############################################
    #         #
    #         #   midpoint curve의 도함수를 구하고
    #         #   추정랜드마크의 r값을 통해 해당 기울기 구함
    #         #   그다음 그거가지고 angle구함
    #         #
    #         #############################################
    #         for i, pnts in enumerate(pred):
    #             midpoints4[i] = np.average(pnts, axis=0)
    #         midpoints_x = midpoints4[:, 1]
    #         der_y = calc_poli(midpoints_x, der_coeff)
    #         atans = np.arctan(der_y) * 180 / np.pi
    #         # 1,-m이 slope vector
    #         slopes = np.ones((17, 2))
    #         slopes[:, 1] = -der_y
    #         angles = _get_angle(slopes, slopes)
    #         pos_p = np.argmax(angles)
    #         pos_p = np.unravel_index(pos_p, angles.shape)
    #         angles_p = angles[pos_p] / np.pi * 180
    #
    #         angles_g, pos_g, _, mid_lines_g, _ = calc_angle_old(gt, (1, 1))
    #         angles_g = angles_g[0]
    #
    #         angle_err = np.abs(angles_g - angles_p) / angles_g
    #
    #         pos_g = pos_g[0:2]
    #         pos_p = pos_p[0:2]
    #
    #         # right = np.tile(lab[:,1,:].expand_dims(1), (1,2,1))
    #         C, H, W = img.shape
    #         plt.figure()
    #
    #         for pos in pos_g:
    #             plt.plot(mid_lines_g[2 * pos:2 * pos + 2, 0] * W, mid_lines_g[2 * pos:2 * pos + 2, 1] * H, 'g')
    #
    #         for pos in pos_p:
    #             # plt.plot(mid_lines_p[2 * pos:2 * pos + 2, 0] * W, mid_lines_p[2 * pos:2 * pos + 2, 1] * H, 'r')
    #             plt.plot([midpoints4[pos, 0] * W], [midpoints4[pos, 1] * H], 'ro', markersize=3.5)
    #
    #         #################################
    #         # fit line 추가
    #         #
    #         #################################
    #         d = 1 / 512
    #         pred = pred.reshape(34, 2, 2)
    #         minyl = pred[0, 0, 1]
    #         minyr = pred[0, 1, 1]
    #         maxyl = pred[33, 0, 1]
    #         maxyr = pred[33, 1, 1]
    #
    #         left_r = np.arange(minyl, maxyl, d)
    #         right_r = np.arange(minyr, maxyr, d)
    #         left_c = calc_poli(left_r, _coeff_l)
    #         right_c = calc_poli(right_r, _coeff_r)
    #         left_line = np.concatenate((left_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)
    #         right_line = np.concatenate((right_c.reshape(-1, 1), right_r.reshape(-1, 1)), axis=1)
    #
    #         mid_c = calc_poli(left_r, coeff)
    #         mid_line = np.concatenate((mid_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)
    #
    #         # plt.plot(left_line[:, 0] * W, left_line[:, 1] * H, color='magenta', alpha=0.5)
    #         # plt.plot(right_line[:, 0] * W, right_line[:, 1] * H, color='magenta', alpha=0.5)
    #
    #         plot_image(img, coord_red=midpoints4.reshape(-1), coord_gr=gt, coord_cy=pred, line_red=mid_line)
    #
    #         # fit line 윤곽추가 맨윗좌표~맨아랫좌표
    #         # title = 'GT%.1f %d %d PR%.1f %d %d ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
    #         #                                                  angles_p, pos_p[0], pos_p[1], angle_err * 100)
    #         title = 'GT%.1f %d %d PR%.1f %.1f %.1f ER%.1f%% ' % (angles_g, pos_g[0], pos_g[1],
    #                                                              angles_p, atans[pos_p[0]], atans[pos_p[1]],
    #                                                              angle_err * 100)
    #         plt.title('{}_R:PRED, G:GT'.format(ind) + '\n' + title)
    #         # plt.show()
    #         save_name = 'way2_{}'.format(ind)
    #         plt.savefig(os.path.join(pred_path, save_name + '.jpg'))
    #         plt.close()
    #
    #         processing_log.append(dict(
    #             pos_GT0=pos_g[0], pos_GT1=pos_g[1], pos_PR0=pos_p[0], pos_PR1=pos_p[1],
    #             angles_g=angles_g, angles_p=angles_p, err=angle_err
    #         ))
    #
    # df = pd.DataFrame(processing_log)
    # df.to_csv(os.path.join(pred_path, save_name + '.csv'))


def postprocess_inte(pred_path, images, labels_gt_abs, title=None, save_plot=False, automatic=False):
    author = 'YB'
    #####   Get absolute pred
    if not os.path.exists(os.path.join(pred_path, 'labels_pred_abs.csv')):
        preds_rel = read_labels(pred_path, title='labels_pred_rel')
        preds = []
        for ind, pred_rel in enumerate(preds_rel):
            _img = images[ind]
            H, W, _ = _img.shape
            pred_abs = to_absolute(label=pred_rel, H=H, W=W)
            preds.append(pred_abs)
        preds = np.asarray(preds)
        write_labels(preds, label_location=pred_path, title='labels_pred_abs')
    else:
        preds = read_labels(pred_path, title='labels_pred_abs')
    preds = label_sort(preds)

    result_list = []
    for ind, image in enumerate(images):
        gt = labels_gt_abs[ind]
        pred = preds[ind]
        H, W, C = image.shape

        fitted_preds, params1 = post_way1(pred, full=True)
        slopes, params2 = post_way2(pred, full=True)

        gt_angles, gt_pos = calc_angle_old(gt, (H, W), full=True)
        w1_angles, w1_pos = calc_angle_old(fitted_preds, (H, W), full=True)
        w2_angles, w2_pos = calc_angle(slopes, get_mid_points(pred),image_H=H, full=True)

        w1_er_3 = np.abs(w1_angles - gt_angles) / (gt_angles + 1e-8)
        w2_er_3 = np.abs(w2_angles - gt_angles) / (gt_angles + 1e-8)

        w1_mainerr = w1_er_3[0]
        w2_mainerr = w2_er_3[0]

        w1_error = np.average(w1_er_3)
        w2_error = np.average(w2_er_3)

        pred = pred.reshape(-1, 2, 2)
        delta = pred[:, 0, :] - pred[:, 1, :]
        deltax = np.average(np.abs(delta)[:, 0]) / W
        # show는 테스트
        if save_plot or automatic:

            #make plot
            plt.figure()
            # R for w1
            left_line = params1['left_line']
            right_line = params1['right_line']
            plt.plot(left_line[:, 0], left_line[:, 1], color='magenta', alpha=0.5)
            plt.plot(right_line[:, 0], right_line[:, 1], color='magenta', alpha=0.5)

            _fp = fitted_preds.reshape(-1,2,2).copy()
            for pos in w1_pos:
                dots = np.average(_fp[2*pos:2*pos+2,:,:], axis = 0)
                plt.plot(dots[:,0], dots[:,1], 'r')

            # blue for w2
            middle_line = params2['line']
            plt.plot(middle_line[:, 0], middle_line[:, 1], color='cyan', alpha=0.5)

            vec_center = params2['middle_vec']
            for pos in w2_pos:
                center = vec_center[pos]
                _slope = slopes[pos]
                norm_slope = _slope / np.linalg.norm(_slope)
                right = center + norm_slope * 50
                left = center - norm_slope * 50
                plt.plot([left[0], right[0]], [left[1], right[1]], color='blue')

            plot_image(image, coord_red=fitted_preds, coord_gr=gt)
            plt_title = 'a %.1f/p1 %.1f/e1 %.2f/p2 %.1f/e2 %.2f'%\
                        (gt_angles[0], w1_angles[0], w1_mainerr, w2_angles[0], w2_mainerr)
            plt.title('test_{}\n'.format(ind) + plt_title)

            if automatic:
                plt.show()
                plt.pause(2)
                plt.close()
            if save_plot:
                if title is None:
                    title = 'pred_{}'.format(ind)
                plt.savefig(os.path.join(pred_path, title + 'png'))
                plt.close()

        res_dict = dict(gt_angle1=gt_angles[0], gt_angle2=gt_angles[1], gt_angle3=gt_angles[2],
                        w1_angle1=w1_angles[0], w1_angle2=w1_angles[1], w1_angle3=w1_angles[2],
                        w2_angle1=w2_angles[0], w2_angle2=w2_angles[1], w2_angle3=w2_angles[2],
                        w1_mainerr=w1_mainerr, w2_mainerr=w2_mainerr,
                        w1_error=w1_error,
                        w2_error=w2_error,
                        avgdx=deltax, gt_pos=gt_pos, w1_pos=w1_pos, w2_pos=w2_pos
                        )
        result_list.append(res_dict)
    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join(pred_path, 'dict_result.csv'))

    return result_list

if __name__ == '__main__':
    plt.ion()
    test_label_location = './test_labels'
    test_data_location = './test_images'
    test_data_names = read_data_names(test_label_location)
    test_labels = read_labels(test_label_location)
    test_images = read_images(test_data_location, test_data_names)

    pred_path = './model/renew_34_nopad_ep2100'
    plt.rcParams["figure.figsize"] = (10, 24)
    postprocess_inte(pred_path = pred_path, images = test_images,
                     labels_gt_abs = test_labels, title=None, save_plot=False, automatic=True)
