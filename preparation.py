#Подготовка изображений
#Переводим всё в 640x640
import os
from pathlib import Path
import cv2
 
directory_img = 'data/images/train'
directory_lbl = 'data/labels/train'
for file in os.listdir(directory_img):
    f = os.path.join(directory_img, file)
    if os.path.isfile(f):
        image = cv2.imread(f)
        h, w = image.shape[:2]
        filename_label = os.path.join(directory_lbl, Path(f).stem+".txt")
        if (h>640)and(w>640):
            kH = 640/h
            kW = 640/w
        text = open(filename_label, 'r+')
        for line in text:
            line_list = line.split(" ")
            print(line_list)
        text.close
 