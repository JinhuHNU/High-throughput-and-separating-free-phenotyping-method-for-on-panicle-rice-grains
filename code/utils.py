import cv2
import numpy as np


def thresh_Seg(img, thresh, type = 3):
    """
    对输入图像进行阈值分割
    :param img: 输入图像，三维彩色图像或者灰度图像
    :param thresh: 分割阈值
    :param type: 默认为3代表输入彩色图像， 若为灰度图需指定为2
    :return: 二值图
    """
    if type == 3:
        gray = img[:, :, 2]
        BN = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    elif type == 2:
        BN = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    return BN

def max_cont_idx(contours):
    """
    找到面积最大的轮廓并返回它的索引值和面积
    :param contours: 所有轮廓
    :return: 面积最大的轮廓的索引和面积
    """
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    max_area = area[max_idx]
    return max_idx, max_area

def filter_dst_cont(contours, thresh):
    """
    输入图像所有轮廓，取出轮廓面积大于某一值的所有轮廓
    :param contours: 由二值图得到的所有轮廓
    :param thresh: 面积阈值，轮廓大于这个值则保留，否着丢弃
    :return: 所有面积大于thresh的轮廓
    """
    cont = []
    for k in range(len(contours)):
        if cv2.contourArea(contours[k]) > thresh:
            cont.append(contours[k])
    return cont

def get_max_contour(img, contours, idx):
    """

    :param img: 输入rgb图像
    :return: 返回一张图像，大小和原图保持一致，但只包括原图中最大轮廓的部分，其它地方置零
    """
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    mask[np.where(mask)] = img[np.where(mask)]
    return mask



def To_Tempalate(img, size=256):
    """
    创建一个size大小的画布，将输入图像放在画布中间
    :param img: rgb图像
    :param size: 画布大小
    :return:
    """
    template = np.zeros((size, size, 3), np.uint8)

    x, y, _ = img.shape
    x1 = size//2 - x//2 - 1 if x % 2 != 0 else size//2 - x//2
    y1 = size//2 - y // 2 - 1 if y % 2 != 0 else size//2 - y // 2
    template[x1:size//2 + x//2, y1:size//2 + y//2] = img
    return template

def Get_W_H(image):
    gray = image[:, :, 2]
    bn = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), dtype=np.uint8)
    bn = cv2.morphologyEx(bn, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_idx, max_area = max_cont_idx(contours)

    rect = cv2.minAreaRect(contours[max_idx])
    perimeter = cv2.arcLength(contours[max_idx], closed=True)
    # print(rect)
    w, h = max(rect[1]), min(rect[1])
    return w, h, max_area, perimeter