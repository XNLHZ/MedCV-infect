import cv2
import os
import data_precut
import shutil
import json


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def img_overlap(img1, img2, label1, label2):
    label = label1 + label2
    height, width = img1.shape[:2]
    for i in range(height):
        for j in range(width):
            img1[i][j][0] = min(img1[i][j][0], img2[i][j][0])
            img1[i][j][1] = min(img1[i][j][1], img2[i][j][1])
            img1[i][j][2] = min(img1[i][j][2], img2[i][j][2])
    return img1, label


