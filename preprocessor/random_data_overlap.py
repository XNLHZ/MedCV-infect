# 小图融合函数与小图镜像函数


import cv2
import os
import data_precut
import shutil
import json
import copy


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def mirror(img, labels, angle):
    # 由于yolo输入数据为640*640，可以支持45°步长的镜像
    height, width = img.shape[:2]
    new_img = copy.deepcopy(img)
    new_labels = copy.deepcopy(labels)

    # 0°镜像
    if angle == 0:
        for i in range(int(height/2)):
            for j in range(width):
                new_img[i][j] = img[height-i-1][j].copy()
                new_img[height-i-1][j] = img[i][j].copy()
        for data in new_labels:
        #     data["points"][0][0] = height - data["points"][0][0] - 1
            data[2] = str(format(1 - float(data[2]), ".6f"))

    # 45°镜像
    if angle == 45:
        for i in range(height):
            for j in range(width-i):
                new_img[i][j] = img[width-j-1][width-i-1].copy()
                new_img[width-j-1][width-i-1] = img[i][j].copy()
        for data in new_labels:
        #     data["points"][0][0], data["points"][0][1] = width - data["points"][0][1]-1, height - data["points"][0][0]-1
            data[1], data[2] = str(format(1 - float(data[2]), ".6f")), str(format(1 - float(data[1]), ".6f"))

    # 90°镜像
    if angle == 90:
        for i in range(height):
            for j in range(int(width/2)):
                new_img[i][j] = img[i][width-j-1].copy()
                new_img[i][width-j-1] = img[i][j].copy()
        for data in new_labels:
        #     data["points"][0][1] = width - data["points"][0][1]-1
            data[1] = str(format(1 - float(data[1]), ".6f"))

    # 135°镜像
    if angle == 135:
        for i in range(height):
            for j in range(i):
                new_img[i][j] = img[j][i].copy()
                new_img[j][i] = img[i][j].copy()
        for data in new_labels:
        #     data["points"][0][0], data["points"][0][1] = data["points"][0][1], data["points"][0][0]
            data[1], data[2] = data[2], data[1]

    return new_img, new_labels



def img_overlap(img1_path, img2_path, label1_path, label2_path, overlapped_data_file, img_index):
    # 默认尺寸相同
    img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)

    # 生成重叠后的json文件
    new_labels = {"shapes": []}
    with open(label1_path, "r", encoding="utf8") as f1:
        for label in json.load(f1)["shapes"]:
            new_labels["shapes"].append(label)
    with open(label2_path, "r", encoding="utf8") as f2:
        for label in json.load(f2)["shapes"]:
            new_labels["shapes"].append(label)
    datas = json.dumps(new_labels, indent=1, ensure_ascii=False)
    with open(overlapped_data_file + "/overlapped_img_" + str(img_index) + ".json", "w", encoding="utf8",
              newline="\n") as fj:
        fj.write(datas)

    # 生成重叠后的图片文件
    height, width = img1.shape[:2]
    for i in range(height):
        for j in range(width):
            img1[i][j][0] = min(img1[i][j][0], img2[i][j][0])
            img1[i][j][1] = min(img1[i][j][1], img2[i][j][1])
            img1[i][j][2] = min(img1[i][j][2], img2[i][j][2])
    cv2.imwrite(overlapped_data_file + "/overlapped_img_" + str(img_index) + ".png", img1)


def main(src_dir, yolo_img_file, yolo_label_file, json_out_file, data_file, overlapped_data_file, train_img_file, train_label_file):
    img_files = copy.deepcopy(os.listdir(yolo_img_file))
    for file in img_files:
        label_data = []
        img = cv2.imread(yolo_img_file + "/" + file)
        with open(yolo_label_file + "/" + file[:-4] + ".txt", "r", encoding="utf8") as f:
            line = f.readline()
            line = line[:-1]
            while line:
                label_data.append(line.split())
                line = f.readline()
                line = line[:-1]
            # print(label_data)
        for i in range(4):
            angle = 45 * i
            new_img, new_label = mirror(img, label_data, angle)

            cv2.imwrite(train_img_file + "/" + file[:-4] + "_" + str(angle) + ".png", new_img)
            with open(train_label_file + "/" + file[:-4] + "_" + str(angle) + ".txt", "w") as f:
                [f.write(" ".join(item) + "\n") for item in new_label]
                f.close()



if __name__ == "__main__":
    # src_dir: 存储原数据的路径
    # out_file: 切割后图片的存储路径
    # yolo_label_file: 切割后yolo格式标注数据的存储路径
    # json_out_file: 切割后各图片,各角度label的原格式数据存储路径（中转数据，验证图片标注点用）
    # data_file: 将原数据存储格式转为函数用格式后的数据路径
    # overlapped_data_file: 重叠后图片的存放路径

    src_dir = "./custom_datas/total_images"
    yolo_img_file = "./custom_datas/total_images"
    yolo_label_file = "./custom_datas/total_labels"
    json_out_file = "./custom_datas/new_json"
    data_file = "./custom_datas/data_file"
    overlapped_data_file = "./custom_datas/overlapped_data_file"
    train_img_file = "./custom_datas/images/train"
    train_label_file = "./custom_datas/labels/train"
    main(src_dir, yolo_img_file, yolo_label_file, json_out_file, data_file, overlapped_data_file, train_img_file, train_label_file)
