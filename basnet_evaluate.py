import argparse
import cv2
import glob
import numpy as np
import os

def calculate_iou(mask_name, gt_mask_root):

    gt_name = mask_name.split('/')[-1]

    pred = cv2.imread(mask_name, 0)
    """
    cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. 
                    It is the default flag. Alternatively, we can pass integer value 1 for this flag.
    cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. 
                    Alternatively, we can pass integer value 0 for this flag.
    cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. 
                    Alternatively, we can pass integer value -1 for this flag.
    """
    pred[pred != 0] = 1

    if os.path.exists(os.path.join(gt_mask_root, )):
        # print(gt_name)
        gt_mask = cv2.imread(os.path.join(gt_mask_root, gt_name), 0)
        gt_mask[gt_mask != 0] = 1

        bitwiseAnd = cv2.bitwise_and(gt_mask, pred)
        bitwiseOr = cv2.bitwise_or(gt_mask, pred)

        # fraction of coverage
        and_area = cv2.countNonZero(bitwiseAnd)
        gt_area = cv2.countNonZero(gt_mask)
        cv_score = float(and_area / (gt_area + 0.000001)) * 100

        # IOU
        or_area = cv2.countNonZero(bitwiseOr)
        iou_score = float(and_area / or_area) * 100
        # print(iou_score)

        return cv_score, iou_score
    else:
        return 0, 0

def compute_average_iou(predict_pth, gt_mask_path):
    cov = []
    iou = []
    pre_list = glob.glob(predict_pth+'/*.png')
    for each in pre_list:
        cov_score, iou_score = calculate_iou(each, gt_mask_path)
        cov.append(cov_score)
        iou.append(iou_score)

    print('the mean cov is: ', np.mean(cov))
    print('the mean iou is: ', np.mean(iou))

    return np.mean(cov), np.mean(iou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_pth', type=str,
                        default='/data/fangcheng.ji/datasets/human_segmentation/test/predicts',
                        help='image dir')
    parser.add_argument('-g', '--gt_mask_path',
                        type=str,
                        default='/data/fangcheng.ji/datasets/human_segmentation/test/masks',
                        help='model path')

    args = parser.parse_args()
    mcov, miou = compute_average_iou(args.predict_pth, args.gt_mask_path)


