import json
import os
import math
import cv2


def format_trans(file_path, out_path):
    label_key = {}
    for file in os.listdir(file_path):
        os.rename(file_path + "/" + file, file_path + "/" + file.replace(" ", ""))
        file = file.replace(" ", "")
        label_list = []
        if file.endswith(".json"):
            with open(file_path + "/" + file, "r", encoding="gbk") as f:
                data = json.load(f)

            for cellData in data["shapes"]:
                if cellData["color"] == 255:
                    continue

                if cellData["color"] not in label_key:
                    label_key[cellData["color"]] = cellData["label"]

                up, down, left, right = 10000, 0, 10000, 0
                # for index, point in enumerate(cellData["contours"]):
                #     cellData["contours"][index] = cellData["contours"][index][0]
                for point in cellData["points"]:
                    width, height = point[0], point[1]
                    if height < up:
                        up = height
                    elif height > down:
                        down = height
                    if width > right:
                        right = width
                    elif width < left:
                        left = width

                label = [cellData["color"], '{:.6f}'.format(((right - left)/2.0 + left) / data["imageWidth"]),
                         '{:.6f}'.format(((down - up)/2.0 + up) / data["imageHeight"]),
                         '{:.6f}'.format((right - left) / data["imageWidth"]),
                         '{:.6f}'.format((down - up) / data["imageHeight"])]
                label_list.append(label)
                # print([cellData['color'], '{:.6f}'.format(((right - left)/2.0)), '{:.6f}'.format(((down - up)/2.0))])
            with open(out_path + "/" + file[:-5] + '.txt', "w") as write_f:
                for label in label_list:
                    write_f.write(' '.join(map(str, label)) + '\n')
    print(label_key)

def format_size(image_path, out_path):
    for image in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image))
        img = cv2.resize(img, (640, 640))
        cv2.imwrite(os.path.join(out_path, image), img)

def main():
    file_path = "C:/Users/XHC/Desktop/医学图像/细胞标注/细胞标注数据/细胞标注_第二次数据/mask/new_mask"
    out_path = "C:/Users/XHC/Desktop/医学图像/细胞标注/细胞标注数据/细胞标注_第二次数据/mask/yolo_label"
    image_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/cell_images'
    image_out_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/format_images'
    format_trans(file_path, out_path)
    # format_size(image_path, image_out_path)


main()