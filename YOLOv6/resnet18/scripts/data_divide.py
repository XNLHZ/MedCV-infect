import cv2
import random
from random import shuffle
import os
import shutil


def data_divide(images, num, img_out_file, train_prop=0.8, val_prop=0.1, test_prop=0.1):
    img_index = []
    for i in range(num):
        img_index.append(i + 1)
    random.shuffle(img_index)
    train_stop = num * train_prop
    val_stop = num * val_prop + train_stop
    test_stop = num * test_prop + val_stop
    count = 0

    destination_img = img_out_file
    # destination_label = label_out_file
    for index in img_index:
        source_img = images + "/nm (" + str(index) + ").jpg"
        # source_label = labels + "img (" + str(index) + ").txt"
        if count < train_stop:
            shutil.copy(source_img, destination_img + "/train/")
            # shutil.copy(source_label, destination_label + "/train/")
            # os.rename(destination_img + "train" + "/sub_img_" + str(index) + ".png",
            #           destination_img + "train" + "/shuffled_sub_img_" + str(count) + ".png")
            # os.rename(destination_label + "train" + "/sub_img_" + str(index) + ".txt",
            #           destination_label + "train" + "/shuffled_sub_img_" + str(count) + ".txt")
        elif count < val_stop:
            shutil.copy(source_img, destination_img + "/val/")
            # shutil.copy(source_label, destination_label + "/val/")
            # os.rename(destination_img + "val" + "/sub_img_" + str(index) + ".png",
            #           destination_img + "val" + "/shuffled_sub_img_" + str(count) + ".png")
            # os.rename(destination_label + "val" + "/sub_img_" + str(index) + ".txt",
            #           destination_label + "val" + "/shuffled_sub_img_" + str(count) + ".txt")
        elif count < test_stop:
            shutil.copy(source_img, destination_img + "/test/")
            # shutil.copy(source_label, destination_label + "test/")
            # os.rename(destination_img + "test" + "/sub_img_" + str(index) + ".png",
            #           destination_img + "test" + "/shuffled_sub_img_" + str(count) + ".png")
            # os.rename(destination_label + "test" + "/sub_img_" + str(index) + ".txt",
            #           destination_label + "test" + "/shuffled_sub_img_" + str(count) + ".txt")
        count += 1


images = "D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/data_nm"              # 存放图片数据的路径
# labels = "D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/cell_labels/"              # 存放标注数据的路径
img_out_file = "/ResNet50-master/cell_data/test"  # 输出划分后图片的路径
# label_out_file = "D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/labels/"   # 输出划分后标注的路径
num = 2853                                         # 图片总数
data_divide(images, num, img_out_file, 0.9, 0, 0.1)

