import copy
import os
import cv2
import math
import json
import random
from shutil import copyfile

def format_trans(label_path, image_path, out_path):
    count_nm, count_sp = 0, 0
    for file in os.listdir(label_path):
        os.rename(label_path + "/" + file, label_path + "/" + file.replace(" ", ""))
        file = file.replace(" ", "")
        if file.endswith(".json"):
            with open(label_path + "/" + file, "r", encoding="utf-8") as f:
                data = json.load(f)

            img = cv2.imread(os.path.join(image_path, file[:-5]+'.jpg'))
            for cellData in data["shapes"]:
                if cellData["label"] == '吞噬':
                    obscure_label = 'special'

                    up = min(int(cellData["points"][0][1]), int(cellData["points"][1][1]))
                    down = max(int(cellData["points"][0][1]), int(cellData["points"][1][1]))
                    left = min(int(cellData["points"][0][0]), int(cellData["points"][1][0]))
                    right = max(int(cellData["points"][0][0]), int(cellData["points"][1][0]))

                    img1 = img[up:down, left:right, ...]
                    if obscure_label == 'special':
                        cv2.imwrite(os.path.join(out_path, 'special', file[:-5] + '-' + str(count_sp) + '.jpg'), img1)
                        count_sp += 1
                    else:
                        continue
    print(count_nm, count_sp)


# def main():
#     label_path = "D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/swallow"
#     image_path = 'D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/swallow'
#     out_path = "D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/tunshi"
#     format_trans(label_path, image_path, out_path)
#     # type_check(file_path)


def filt(data_path, out_path):
    list = []
    while len(list) < 250:
        i = random.randint(1, 2853)
        if i not in list:
            list.append(i)
    for idx in list:
        source_file = os.path.join(data_path, 'nm (' + str(idx) + ').jpg')
        destination_file = os.path.join(out_path, 'nm (' + str(idx) + ').jpg')
        copyfile(source_file, destination_file)


def getRotatedImg(angle, img):
    rows, cols = img.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D((a, b), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))  # 旋转后的图像保持大小不变
    return rotated_img


def mirror(img, angle):
    # 由于yolo输入数据为640*640，可以支持45°步长的镜像
    height, width = img.shape[:2]
    new_img = copy.deepcopy(img)

    # 0°镜像
    if angle == 0:
        for i in range(int(height/2)):
            for j in range(width):
                new_img[i][j] = img[height-i-1][j].copy()
                new_img[height-i-1][j] = img[i][j].copy()

    # 45°镜像
    if angle == 45:
        for i in range(height):
            for j in range(width-i):
                new_img[i][j] = img[width-j-1][width-i-1].copy()
                new_img[width-j-1][width-i-1] = img[i][j].copy()

    # 90°镜像
    if angle == 90:
        for i in range(height):
            for j in range(int(width/2)):
                new_img[i][j] = img[i][width-j-1].copy()
                new_img[i][width-j-1] = img[i][j].copy()

    # 135°镜像
    if angle == 135:
        for i in range(height):
            for j in range(i):
                new_img[i][j] = img[j][i].copy()
                new_img[j][i] = img[i][j].copy()

    return new_img


def augument(data_path):
    for image in os.listdir(data_path):
        if image.endswith('jpg'):
            img = cv2.imread(os.path.join(data_path, image))
            rows, cols = img.shape[:2]
            ##填充图像为正方形，而且要能保证填充后的图像在0到360°旋转的时候，原图像的像素不会损失
            if rows > cols:
                re = cv2.copyMakeBorder(img, 0, 0, int((rows-cols) / 2), int(rows-cols-int((rows-cols) / 2)), cv2.BORDER_CONSTANT, value=(255,255,255))

            for angle1 in range(0, 180, 90):
                img = getRotatedImg(angle1, re)

                for angle2 in range(0, 180, 45):
                    img = mirror(img, angle2)
                    cv2.imwrite(os.path.join(data_path, 'augument_data', image[:-4] + '_' + str(angle1) + '_' + str(angle2) + '.jpg'), img)


if __name__ == '__main__':
    data_path = 'D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/data'
    out_path = 'D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/train'
    filt(data_path, out_path)