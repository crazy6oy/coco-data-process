# ################################################
# 目标：使用匈牙利算法根据两个样本中的IoU配对。
# 明确概念：开销方阵/矩阵、优势方阵/矩阵。
#   优势方阵/矩阵（基于IoU举例）：每个方格中的内容是横纵行对应目标的交并比；
#   开销方阵/矩阵（基于IoU举例）：1 - 优势方阵/矩阵。
# 匈牙利算法基于开销方阵计算配对。
#
# 输入coco annFile（）格式的json文件
#
# ################################################

import cv2
import numpy as np
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment


def coco_format_to_image_ann(ann_file):
    image_id_name = dict()
    for img in ann_file.imgs.values():
        image_id_name[img["id"]] = img["file_name"]

    category_id_name = dict()
    for cats in ann_file.cats.values():
        category_id_name[cats["id"]] = cats["name"]

    image_ann = dict()
    for ann in ann_file.anns.values():
        image_id = ann["image_id"]
        image_name = image_id_name[image_id]
        category_id = ann["category_id"]
        category_name = category_id_name[category_id]
        category_prefix, category_suffix = category_name.split("-")
        # category_prefix, category_suffix = category_name, 1
        segmentation = ann["segmentation"]
        if image_name not in image_ann.keys():
            image_ann[image_name] = {}
        if category_prefix not in image_ann[image_name].keys():
            image_ann[image_name][category_prefix] = {}
        if category_suffix not in image_ann[image_name][category_prefix].keys():
            image_ann[image_name][category_prefix][category_suffix] = []
        image_ann[image_name][category_prefix][category_suffix].append(segmentation)

    return image_ann


def dict_key_intersection(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    return sorted(keys1 & keys2)


def black_draw_polygon(img_w, img_h, polygons):
    white = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(white, np.array(poly, np.int16), 255)

    return white


def draw_obj_id(img_h, img_w, img_cat_obj):
    black = np.zeros((img_h, img_w), dtype=np.uint8)
    for object_id in img_cat_obj.keys():
        cv2.fillPoly(black, [np.array(x, dtype=np.int32).reshape((-1, 2)) for x in img_cat_obj[object_id]], object_id)

    return black


def obj_mask_to_iou_matrix(mask1, mask2):
    h_max = np.max(mask1)
    w_max = np.max(mask2)

    matrix = np.zeros((h_max, w_max))
    for x in range(1, w_max + 1):
        for y in range(1, h_max + 1):
            matrix[y - 1, x - 1] = np.sum((mask1 == x) & (mask2 == y)) / np.sum((mask1 == x) | (mask2 == y))

    return matrix


def box_relationship(box1, box2):
    """
    :param box1: [[xl,yu], [xr,yd]]
    :param box2: [[xl,yu], [xr,yd]]
    :return: 0相离, 1相交
    """
    min_x = min(box1[0][0], box2[0][0])
    min_y = min(box1[0][1], box2[0][1])
    max_x = max(box1[1][0], box2[1][0])
    max_y = max(box1[1][1], box2[1][1])
    u_w = max_x - min_x
    u_h = max_y - min_y

    box1_w = box1[1][0] - box1[0][0]
    box1_h = box1[1][1] - box1[0][1]
    box2_w = box2[1][0] - box2[0][0]
    box2_h = box2[1][1] - box2[0][1]

    if u_w > (box1_w + box2_w) or u_h > (box1_h + box2_h):
        return 0
    else:
        return 1


def obj_iou(box1, box2):
    """
    :param box1: [[xl,yu], [xr,yd]]
    :param box2: [[xl,yu], [xr,yd]]
    :return: iou
    """

    if box_relationship(box1, box2) == 0:
        return 0

    inter_w = min(abs(box1[1][0] - box2[0][0]), abs(box2[1][0] - box1[0][0]))
    inter_h = min(abs(box1[1][1] - box2[0][1]), abs(box2[1][1] - box1[0][1]))
    union_w = max(abs(box1[1][0] - box2[0][0]), abs(box2[1][0] - box1[0][0]))
    union_h = max(abs(box1[1][1] - box2[0][1]), abs(box2[1][1] - box1[0][1]))

    return (inter_w * inter_h) / (union_w * union_h)


def obj_diou(box1, box2):
    """
    :param box1: [[xl,yu], [xr,yd]]
    :param box2: [[xl,yu], [xr,yd]]
    :return: diou
    """

    union_w = max(abs(box1[1][0] - box2[0][0]), abs(box2[1][0] - box1[0][0]))
    union_h = max(abs(box1[1][1] - box2[0][1]), abs(box2[1][1] - box1[0][1]))
    center_w = abs((box1[1][0] + box1[0][0]) / 2 - (box2[1][0] + box2[0][0]) / 2)
    center_h = abs((box1[1][1] + box1[0][1]) / 2 - (box2[1][1] + box2[0][1]) / 2)

    return obj_iou(box1, box2) - (center_w ** 2 + center_h ** 2) / (union_w ** 2 + union_h ** 2)


def obj_ciou(box1, box2):
    pass


def main():
    height = 1080
    weight = 1920
    mask_iou_threshold = 0.8
    obj_diou_threshold = 0.8

    ann_file1 = COCO(r"C:\Users\wangyx\Desktop\images\label\annotations.json")
    ann_file2 = COCO(r"C:\Users\wangyx\Desktop\images\label\annotations.json")

    image_ann_er1 = coco_format_to_image_ann(ann_file1)
    image_ann_er2 = coco_format_to_image_ann(ann_file2)

    intersection_images_name = dict_key_intersection(image_ann_er1, image_ann_er2)
    for image_name in intersection_images_name:
        intersection_category_name = dict_key_intersection(image_ann_er1[image_name], image_ann_er2[image_name])
        for category_name in intersection_category_name:
            anner1_mask = draw_obj_id(height, weight, image_ann_er1[image_name][category_name])
            anner2_mask = draw_obj_id(height, weight, image_ann_er2[image_name][category_name])

            cost_matrix = obj_mask_to_iou_matrix(anner1_mask, anner2_mask)
            if cost_matrix.shape[0] != cost_matrix.shape[1]:
                print("图片{}的{}类别两个人标注对向数不对应，请检查目标数".format(image_name, category_name))
                continue
            rows_ind, cols_ind = linear_sum_assignment(1 - cost_matrix)
            for row_ind, col_ind in zip(rows_ind, cols_ind):
                obj_iou = cost_matrix[row_ind, col_ind]
                if obj_iou < mask_iou_threshold:
                    print("图片{}的{}类别第一个人的目标{}和第二个人的目标{},分割IoU较低:{}, 请检查标注范围".format(
                        image_name,
                        category_name,
                        row_ind,
                        col_ind,
                        obj_iou))
                    continue

                obj1_nonzero_ind = np.nonzero(anner1_mask == (row_ind + 1))
                box1_min_x = np.min(obj1_nonzero_ind[0])
                box1_min_y = np.min(obj1_nonzero_ind[1])
                box1_max_x = np.max(obj1_nonzero_ind[0])
                box1_max_y = np.max(obj1_nonzero_ind[1])
                box1 = [[box1_min_x, box1_min_y], [box1_max_x, box1_max_y]]
                obj2_nonzero_ind = np.nonzero(anner2_mask == (col_ind + 1))
                box2_min_x = np.min(obj2_nonzero_ind[0])
                box2_min_y = np.min(obj2_nonzero_ind[1])
                box2_max_x = np.max(obj2_nonzero_ind[0])
                box2_max_y = np.max(obj2_nonzero_ind[1])
                box2 = [[box2_min_x, box2_min_y], [box2_max_x, box2_max_y]]
                diou = obj_diou(box1, box2)
                if diou < obj_diou_threshold:
                    print("图片{}的{}类别第一个人的目标{}和第二个人的目标{}分割IoU达标, 但外接框IoU:{}，请检查是否有离散区域影响".format(
                        image_name,
                        category_name,
                        row_ind,
                        col_ind,
                        diou))
                stop = 0


if __name__ == '__main__':
    main()
