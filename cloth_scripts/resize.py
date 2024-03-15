import os
import cv2
import numpy as np


def resize(file_path, label_path, img_out_path, label_out_path, w=640, h=640):
    for file in os.listdir(file_path):
        if file.endswith('jpg'):
            # 图片resize前置
            img = cv2.imread(os.path.join(file_path, file))
            height, width, c = img.shape
            m = int(640.0 / width * height)  # 缩放后的长度
            img_new = np.zeros([640, 640, 3], dtype=np.uint8)
            img1 = cv2.resize(img, (640, m), interpolation=cv2.INTER_AREA)
            start = int((640 - m) / 2)
            end = start + m

            # 标签resize
            file_data = ""
            flag = 1
            if os.path.exists(os.path.join(label_path, file[:-3] + 'txt')):
                with open(os.path.join(label_path, file[:-3] + 'txt'), 'r') as f:
                    flag = 1
                    for line in f:
                        flag = 0
                        new_line = label_resize(line, start, m)
                        line = line.replace(line, new_line)
                        file_data += line + '\n'
                if flag:        # 剔除无标签图片
                    continue
                with open(os.path.join(label_out_path, file[:-3] + 'txt'), "w", encoding="utf-8") as f:
                    f.write(file_data)

            # 图片resize
            for row in range(0, 640):
                for col in range(640):
                    if row < start:
                        img_new[row][col] = np.array([255, 255, 255])
                    elif row < end:
                        img_new[row][col] = img1[row-start][col]
                    else:
                        img_new[row][col] = np.array([255, 255, 255])

            cv2.imwrite(os.path.join(img_out_path, file), img_new)





def label_resize(old_line, start, m):
    # 仅适用于宽比长大的图片
    line = old_line.split(' ')
    line[2] = str((start + int(float(line[2]) * m)) / 640.0)
    line[4] = str(int(float(line[4]) * m) / 640.0)
    line = ' '.join(line)
    return line


def point_mark(img_path, label_path, out_path):
    color = (0, 0, 255)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip('/')
            line = line.split(' ')
            x1 = int((float(line[1]) - float(line[3]) / 2.0) * width)
            y1 = int((float(line[2]) - float(line[4]) / 2.0) * height)
            x2 = int((float(line[1]) + float(line[3]) / 2.0) * width)
            y2 = int((float(line[2]) + float(line[4]) / 2.0) * height)
            p1 = (x1, y1)
            p2 = (x2, y2)
            cv2.rectangle(img, p1, p2, color)
    cv2.imwrite(out_path, img)


if __name__ == '__main__':
    file_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/cutted'
    label_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test'
    img_out_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/resize'
    label_out_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/resized/labels'
    resize(file_path, label_path, img_out_path, label_out_path)

    img_path1 = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test'
    label_path1 = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test'
    out_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/resized'
    # point_mark(img_path1, label_path1, out_path+'/img1.jpg')