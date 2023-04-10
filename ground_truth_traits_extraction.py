
import cv2
from cv2 import imread
import numpy as np
import os
import csv

img_dir = r"C:\Users\DELL\Desktop\data\kendao1867"
for i in range(20):

    img_name = r"kd1867_{}_3.tif".format(i+1)
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    gray = img[:, :, 2]
    bn = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(bn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = []        #Grian length list
    width = []         #Grian width list
    areas = []         #Grain project area list
    perimeter = []     #Grain perimeter list
    lw_ratio = []      #Grain length/width ratio list
    ap_ratio = []      #Grain area/perimeter ratio list
    circularity = []   #Grain circularity list

    for item in contours:
        area = cv2.contourArea(item)
        if area >=1000:
            rect = cv2.minAreaRect(item)
            length.append(max(rect[1])*0.042333)
            width.append(min(rect[1])*0.042333)
            areas.append(area*0.042333*0.042333)
            perimeter.append(cv2.arcLength(item, closed=True)*0.042333)
            lw_ratio.append(max(rect[1])/min(rect[1]))
            ap_ratio.append((area*0.042333*0.042333)/(cv2.arcLength(item, closed=True)*0.042333))
            circularity.append(4*np.pi*area/np.power(cv2.arcLength(item, closed=True), 2))

    avg_length = np.average(length)
    avg_width = np.average(width)
    avg_area = np.average(areas)
    avg_peri = np.average(perimeter)
    avg_lw_ratio = np.average(lw_ratio)
    avg_ap_ratio = np.average(ap_ratio)
    avg_circularity = np.average(circularity)
    std_length = np.std(length)
    std_width = np.std(width)
    std_area = np.std(areas)
    std_perimeter = np.std(perimeter)
    std_lw_ratio = np.std(lw_ratio)
    std_ap_ratio = np.std(ap_ratio)
    std_circularity = np.std(circularity)

  
    row = (img_name, len(length), avg_length, avg_width, avg_peri, avg_area, avg_lw_ratio, avg_ap_ratio, avg_circularity, std_length, std_width, std_area, std_perimeter, std_lw_ratio, std_ap_ratio, std_circularity)

    with open('ground_truth_panicle_scalr.csv', 'a', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)


