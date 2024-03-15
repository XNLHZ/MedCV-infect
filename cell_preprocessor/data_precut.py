import cv2
import os
import json
import numpy as np
import math
from math import *
import shutil

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# 计算点间距
def d_cal(label):
    x1 = label["points"][0][0]
    y1 = label["points"][0][1]
    x2 = label["points"][1][0]
    y2 = label["points"][1][1]

    diameter = 2 * sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))
    return format(diameter, ".6f")


# 小图到大图的坐标转换
def cdt_s2b(x1, y1, x0, y0, angle):
    if y0 == 0:
        angle_img = 90
    # x0,y0表示标注点在小图中的坐标
    # x1,y1表示小图左上角在原图中的坐标
    # x2,y2表示标注点在原图中的坐标
    else:
        angle_img = atan(x0 / y0) / math.pi * 180
    angle_total = angle_img + angle
    diagonal_len = math.sqrt(pow(x0, 2) + pow(y0, 2))
    angle_total = angle_total / 180 * math.pi
    x2 = x1 + diagonal_len * sin(angle_total)
    y2 = y1 + diagonal_len * cos(angle_total)
    return x2, y2

# 大图到小图的坐标转换
def cdt_b2s(x1, y1, x2, y2, angel):
    if y2 - y1 == 0:
        angle_total = 90
    else:
        angle_total = atan((x2 - x1) / (y2 - y1)) / math.pi * 180
    if angle_total < 0:
        angle_total += 180
    angel_img = angle_total - angel
    diagonal_len = math.sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))
    angel_img = angel_img / 180 * math.pi
    x0 = diagonal_len * sin(angel_img)
    y0 = diagonal_len * cos(angel_img)
    return x0, y0

# 计算边界
def cal_bound(pre_width, pre_height, width, height, angle):
    angle = angle / 180 * math.pi
    bound_width_right = pre_width - width * cos(angle)
    bound_width_left = height * sin(angle)
    bound_height_top = 0
    bound_height_bottom = pre_height - width * sin(angle) - height * cos(angle)
    bound = [bound_width_left, bound_width_right, bound_height_top, bound_height_bottom]
    return bound

# 生成旋转小图
def img_generate(source_img, x1, y1, width, height, angle):
    img = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            x2, y2 = cdt_s2b(x1, y1, i, j, angle)
            x2, y2 = int(x2), int(y2)
            img[i][j] = source_img[x2][y2]
    return img


def point_location(x1, y1, x2, y2, x0, y0):
    A = x1 - x0
    B = y2 - y0
    C = y1 - y0
    D = x2 - x0
    return A * B - C * D

# 某点是否包含于某图内
def judge_include(x1, y1, x0, y0, angle, width, height):
    angle = angle / 180 * math.pi
    x2 = x1 + width * sin(angle)
    y2 = y1 + width * cos(angle)
    x3 = x1 + height * cos(angle)
    y3 = y1 - height * sin(angle)
    x4 = x3 + width * sin(angle)
    y4 = y3 + width * cos(angle)
    if point_location(x1, y1, x2, y2, x0, y0) * point_location(x3, y3, x4, y4, x0, y0) > 0:
        return False
    if point_location(x1, y1, x3, y3, x0, y0) * point_location(x2, y2, x4, y4, x0, y0) > 0:
        return False
    return True

# 展示标注点
def point_mark(img, json_datas, index="", out_file="mark_images"):
    color = (0, 0, 255)
    for data in json_datas:
        x, y = int(data["points"][0][0]), int(data["points"][0][1])
        cv2.circle(img, (y, x), 5, color, 1)
    cv2.imwrite(out_file + "/" + "mark_img" + index + ".png", img)

# 标签转换
def label_transform(data_file, images_after_cut, js, width, height, num, pre_angle, page_index, json_out_file):
    new_labels = {}
    with open(data_file + "/" + js, "r", encoding="utf8") as fp:
        json_datas = json.load(fp)["shapes"]
        for i, img_data in enumerate(images_after_cut):
            x1, y1, angle = img_data[0][0], img_data[0][1], img_data[0][2]
            new_labels[str(num + i)] = new_labels.get(i, [])
            for data in json_datas:
                x2, y2 = data["points"][0][1], data["points"][0][0]
                # x2_1, y2_1 = data["points"][1][1], data["points"][1][0]
                x0, y0 = cdt_b2s(x1, y1, x2, y2, angle)
                # x0_1, y0_1 = cdt_b2s(x1, y1, x2_1, y2_1, angle)
                if judge_include(x1, y1, x2, y2, angle, width, height):
                    new_label = {"label": data["label"], "points": [[x0, y0]], "group_id": data["group_id"],
                                 "shape_type": data["shape_type"], "flags": data["flags"],
                                 'cell_width': data['cell_width'], 'cell_height': data['cell_height']}
                    new_labels[str(num + i)].append(new_label)

    # for key in sorted(new_labels.keys()):
    #     print(str(key) + ": ")
    #     for label in new_labels[key]:
    #         print(label)

    data = json.dumps(new_labels, indent=1, ensure_ascii=False)
    with open(json_out_file + "/" + "new_json_" + str(page_index) + "_" + str(pre_angle) + ".json", "w",
              encoding="utf8", newline="\n") as f2:
        f2.write(data)
    new_json_file = "new_json_" + str(page_index) + "_" + str(pre_angle) + ".json"
    return new_json_file

# 标注点可视化
def cut_img_mark(images_after_cut, js):
    with open(js, "r", encoding="utf8") as fp:
        json_datas_sum = json.load(fp)
    for i, img_data in enumerate(images_after_cut):
        img = img_data[1]
        index = i
        json_data = json_datas_sum[str(i)]
        point_mark(img, json_data, str(index), out_file="mark_images")

# json转yolo
def yolo_data_trans(images_after_cut, new_json_file, yolo_label_file, json_out_file):
    bbox_width, bbox_height = None, None
    height, width = images_after_cut[0][1].shape[:2]
    with open(json_out_file + "/" + new_json_file, "r", encoding="utf8") as fp:
        json_datas = json.load(fp)
        for key in sorted(json_datas.keys()):
            yolo_txt = []
            for label in json_datas[key]:
                # diameter = d_cal(label)
                bbox_width, bbox_height = float(label['cell_width']), float(label['cell_height'])
                if label["label"] == "中性粒细胞":
                    label_type = "0"
#                    bbox_width, bbox_height = 10, 10
                elif label["label"] == '单核巨噬细胞':
                    label_type = "1"
#                    bbox_width, bbox_height = 30, 30
                elif label["label"] == "淋巴细胞":
                    label_type = "2"
#                    bbox_width, bbox_height = 30, 30
                prop_point_width = format(label["points"][0][1] / width, ".6f")
                prop_point_height = format(label["points"][0][0] / height, ".6f")
                prop_bbox_width = format(bbox_width / width, ".6f")
                prop_bbox_height = format(bbox_height / height, ".6f")
                yolo_label = label_type + " " + str(prop_point_width) + " " + str(prop_point_height) \
                             + " " + str(prop_bbox_width) + " " + str(prop_bbox_height)
                yolo_txt.append(yolo_label)
            with open(yolo_label_file + "/" + "sub_img_" + str(key) + ".txt", "w") as f:
                [f.write(item + "\n") for item in yolo_txt]
                f.close()

# 直角旋转
def rect_rotate(img, rotate_angle):
    # print(type(img))
    height, width = img.shape[:2]
    new_img = None
    if rotate_angle == 0:
        return img
    elif rotate_angle == 270:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[j][height - i - 1] = img[i][j]
    elif rotate_angle == 180:
        new_img = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[height - i - 1][width - j - 1] = img[i][j]
    elif rotate_angle == 90:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[width - j - 1][i] = img[i][j]
    return new_img

# 角度设定转换
def angle_trans(img, angle, js, data_file):
    rotate_angle = 0
    new_angle = angle
    if 90 < angle <= 180:
        rotate_angle = 90
        new_angle = angle - 90
    elif 180 < angle <= 270:
        rotate_angle = 180
        new_angle = angle - 180
    elif 270 < angle <= 360:
        rotate_angle = 270
        new_angle = angle - 270
    # print(type(img))
    new_img = rect_rotate(img, rotate_angle)
    new_js = json_cdt_rotate(js, img, rotate_angle, data_file)
    return new_img, new_angle, new_js

# json文件坐标旋转
def json_cdt_rotate(js, img, rotate_angle, data_file):
    x2, y2 = 0, 0
    new_labels = {"shapes": []}
    height, width = img.shape[:2]
    with open(data_file + "/" + js, "r", encoding="gbk") as fp:
        json_datas = json.load(fp)["shapes"]
    for data in json_datas:
        x2, y2 = data["points"][0][1], data["points"][0][0]
        # x2_1, y2_1 = data["points"][1][1], data["points"][1][0]
        if rotate_angle == 0:
            pass
        elif rotate_angle == 90:
            x2, y2 = width - y2 - 1, x2
            # x2_1, y2_1 = width - y2_1 - 1, x2_1
        elif rotate_angle == 180:
            x2, y2 = height - x2 - 1, width - y2 - 1
            # x2_1, y2_1 = height - x2_1 - 1, width - y2_1 - 1
        elif rotate_angle == 270:
            x2, y2 = y2, height - x2 - 1
            # x2_1, y2_1 = y2_1, height - x2_1 - 1
        new_label = {"label": data["label"], "points": [[y2, x2]], "group_id": data["group_id"],
                     "shape_type": data["shape_type"], "flags": data["flags"],
                     'cell_width': data['cell_width'], 'cell_height': data['cell_height']}
        new_labels["shapes"].append(new_label)
    datas = json.dumps(new_labels, indent=1, ensure_ascii=False)
    with open(data_file + "/" + "rotated_json.json", "w", encoding="utf8", newline="\n") as f2:
        f2.write(datas)
    rotated_json_file = "rotated_json.json"
    return rotated_json_file

#
def crop_json(data_file, js, img_path, angle, width=640, height=480, step_width=640, step_height=480, out_file=None,
              yolo_label_file=None, num=0, page_index=0, json_out_file="new_json"):
    img = cv2.imread(data_file + "/" + img_path)
    images_after_cut = []
    pre_angle = angle
    # print(type(img))
    img, angle, js = angle_trans(img, angle, js, data_file)
    pre_height, pre_width = img.shape[:2]
    bound = cal_bound(pre_width, pre_height, width, height, angle)
    now_x1, now_y1 = 0, bound[0]
    while now_x1 < int(bound[3]):
        while now_y1 < int(bound[1]):
            images_after_cut.append(
                [[now_x1, now_y1, angle], img_generate(img, now_x1, now_y1, width, height, angle)])
            now_y1 += step_width
        now_x1 += step_height
        now_y1 = bound[0]

    for i, img_data in enumerate(images_after_cut):
        img = img_data[1]
        cv2.imwrite(out_file + "/" + "sub_img_" + str(num + i) + ".png", img)

    new_json_file = label_transform(data_file, images_after_cut, js, width, height, num, pre_angle, page_index,
                                    json_out_file)
    # cut_img_mark(images_after_cut, new_json_file)
    yolo_data_trans(images_after_cut, new_json_file, yolo_label_file, json_out_file)

    return num + len(images_after_cut)

#
def train_data_generate(img_path_and_js, out_file, yolo_label_file, json_out_file, data_file, num, page_index):
    for path in img_path_and_js:
        img_path = path[0]
        js = path[1]
        for angle in range(12):
            num = crop_json(data_file, js, img_path, angle * 30, 640, 640, 450, 450, out_file, yolo_label_file, num,
                            page_index, json_out_file)
        page_index += 1

#
def main(src_dir, out_file, yolo_label_file, json_out_file, data_file):
    check_dir(out_file)
    check_dir(yolo_label_file)
    check_dir(json_out_file)
    check_dir(data_file)
    imgs, jsons, img_path_and_js = [], [], []
    for file in os.listdir(src_dir):
        # for file in os.listdir(os.path.join(src_dir, bac_dir)):
        if not os.path.exists(data_file + "/" + file.replace(' ', '')):
            shutil.copy(src_dir + "/" + file, data_file)
            os.rename(data_file + "/" + file, data_file + "/" + file.replace(' ', ''))
        file = file.replace(' ', '')
        if file.endswith('jpg'):
            imgs.append(file)
        if file.endswith('json'):
            jsons.append(file)
    imgs.sort()
    jsons.sort()
    for i, img in enumerate(imgs):
        img_path_and_js.append([img, jsons[i]])
    num, page_index = 0, 1
    train_data_generate(img_path_and_js, out_file, yolo_label_file, json_out_file, data_file, num, page_index)


#
if __name__ == "__main__":
    # src_dir: 存储原数据的路径
    # out_file: 切割后图片的存储路径
    # yolo_label_file: 切割后yolo格式标注数据的存储路径
    # json_out_file: 切割后各图片,各角度label的原格式数据存储路径（中转数据，验证图片标注点用）
    # data_file: 将原数据存储格式转为函数用格式后的数据路径

    src_dir = "D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/old_json_data_test"
    out_file = "D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/images/test"
    yolo_label_file = "D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/labels/test"
    json_out_file = "D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/temp_new_json"
    data_file = "D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/data_file_test"
    main(src_dir, out_file, yolo_label_file, json_out_file, data_file)
