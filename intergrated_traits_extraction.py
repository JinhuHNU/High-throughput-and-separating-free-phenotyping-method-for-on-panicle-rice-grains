import cv2
import os
import numpy as np
import utils

length = []        #Grian length list
width = []         #Grian width list
area = []          #Grian project area list
perimeter = []     #Grian perimeter list
lw_ratio = []      #Grian length/width ratio list
ap_ratio = []      #Grian area/perimeter ratio list
circularity = []   #Grian circularity list

img_dir = r"I:\recovery_intact\jjxx\16"
name1 = os.path.split(img_dir)
name2 = os.path.split(name1[0])
name = name2[1]+"_"+name1[1]

img_list = os.listdir(img_dir)
num = len(img_list)
for i, item in enumerate(img_list):
    img_path = os.path.join(img_dir, item)
    img = cv2.imread(img_path)
    recovery = img[:, 256:, :]
    bn = utils.thresh_Seg(recovery, 60)
    contours, _ = cv2.findContours(bn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    panicle = utils.filter_dst_cont(contours, 1000)
    for k, whatever in enumerate(panicle):
        area.append(cv2.contourArea(whatever)*0.042333*0.042333)
        rect = cv2.minAreaRect(whatever)
        # print(rect)
        l, w = max(rect[1]), min(rect[1])
        length.append(l*0.042333)
        width.append(w*0.042333)
        perimeter.append(cv2.arcLength(whatever, closed=True)*0.042333)
        lw_ratio.append((l*0.042333) / (w*0.042333))
        ap_ratio.append((cv2.contourArea(whatever)*0.042333*0.042333)/(cv2.arcLength(whatever, closed=True)*0.042333))
        circularity.append(4*np.pi*(cv2.contourArea(whatever)*0.04233*0.04233)/np.power((cv2.arcLength(whatever, closed=True)*0.042333), 2))

avg_length = np.average(length)
avg_width = np.average(width)
avg_area = np.average(area)
avg_peri = np.average(perimeter)
avg_lw_ratio = np.average(lw_ratio)
avg_ap_ratio = np.average(ap_ratio)
avg_circularity = np.average(circularity)
std_length = np.std(length)
std_width = np.std(width)
std_area = np.std(area)
std_perimeter = np.std(perimeter)
std_lw_ratio = np.std(lw_ratio)
std_ap_ratio = np.std(ap_ratio)
std_circularity = np.std(circularity)

row = (name, num, avg_length, avg_width, avg_peri, avg_area, avg_lw_ratio, avg_ap_ratio, avg_circularity, 
    std_length, std_width, std_perimeter, std_area, std_lw_ratio, std_ap_ratio, std_circularity)

import csv

# open the file in the write mode
with open('result_test.csv', 'a', encoding='UTF8', newline="") as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(row)