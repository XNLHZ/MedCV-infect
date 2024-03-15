import os
import cv2
import json

def data_cut(img_dir, json_label_dir, img_output_dir, label_output_dir):
    for json_file in os.listdir(json_label_dir):
        with open(os.path.join(json_label_dir, json_file), "r", encoding="utf8") as fp:
            json_datas = json.load(fp)
            height, width = 1000, 2446
            m = int(width / 2)
            for label in json_datas:
                yolo_txt1 = []
                yolo_txt2 = []
                # 分割图片
                img = cv2.imread(os.path.join(img_dir, label['name']))
                height, width = img.shape[:2]
                flag = [0, 0]
                img1 = img[:, 0:int(width * 0.5)]
                img2 = img[:, int(width * 0.5):int(width)]

                # 分割标签

                xmin = int(label['bbox'][0])
                ymin = int(label['bbox'][1])
                xmax = int(label['bbox'][2])
                ymax = int(label['bbox'][3])
                # 判断在左半边还是右半边
                # 右半边
                if xmin >= m:
                    flag[1] = 1
                    img2_x1 = xmin - m
                    img2_y1 = ymin
                    img2_x2 = xmax - m
                    img2_y2 = ymax
                    yolo_label2 = ['0', str(((img2_x2 - img2_x1) / 2.0 + img2_x1) / (width / 2)),
                                  str(((img2_y2 - img2_y1) / 2.0 + img2_y1) / height),
                                  str((img2_y2 - img2_y1) / width * 2),
                                  str((img2_y2 - img2_y1) / height)]
                    str_label = ' '.join(yolo_label2)
                    yolo_txt2.append(str_label)
                # 左半边
                elif xmax <= m:
                    flag[0] = 1
                    img1_x1 = xmin
                    img1_y1 = ymin
                    img1_x2 = xmax
                    img1_y2 = ymax
                    yolo_label1 = ['0', str(((img1_x2 - img1_x1) / 2.0 + img1_x1) / width * 2),
                                   str(((img1_y2 - img1_y1) / 2.0 + img1_y1) / height),
                                   str((img1_y2 - img1_y1) / width * 2),
                                   str((img1_y2 - img1_y1) / height)]
                    str_label = ' '.join(yolo_label1)
                    yolo_txt1.append(str_label)
                # 中间
                else:
                    flag = [1, 1]
                    img1_x1 = xmin
                    img1_y1 = ymin
                    img1_x2 = m
                    img1_y2 = ymax
                    yolo_label1 = ['0', str(((img1_x2 - img1_x1) / 2.0 + img1_x1) / width * 2),
                                   str(((img1_y2 - img1_y1) / 2.0 + img1_y1) / height),
                                   str((img1_y2 - img1_y1) / width * 2),
                                   str((img1_y2 - img1_y1) / height)]
                    str_label = ' '.join(yolo_label1)
                    yolo_txt1.append(str_label)

                    img2_x1 = m
                    img2_y1 = ymin
                    img2_x2 = xmax - m
                    img2_y2 = ymax
                    yolo_label2 = ['0', str(((img2_x2 - img2_x1) / 2.0 + img2_x1) / width * 2),
                                   str(((img2_y2 - img2_y1) / 2.0 + img2_y1) / height),
                                   str((img2_y2 - img2_y1) / width * 2),
                                   str((img2_y2 - img2_y1) / height)]
                    str_label = ' '.join(yolo_label2)
                    yolo_txt2.append(str_label)

                if flag[0] == 1:
                    cv2.imwrite(os.path.join(img_output_dir, label['name'][:-4] + '_1.jpg'), img1)
                    with open(os.path.join(label_output_dir, label['name'][:-4] + '_1.txt'), 'w') as f:
                        f.writelines(yolo_txt1)
                if flag[1] == 1:
                    cv2.imwrite(os.path.join(img_output_dir, label['name'][:-4] + '_2.jpg'), img2)
                    with open(os.path.join(label_output_dir, label['name'][:-4] + '_2.txt'), 'w') as f:
                        f.writelines(yolo_txt2)


def img_cut(img_dir, img_output_dir):
    for file in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, file))
        height, width = img.shape[:2]
        img1 = img[:, 0:int(width * 0.5)]
        img2 = img[:, int(width * 0.5):int(width)]
        cv2.imwrite(os.path.join(img_output_dir, file[:-4] + '_1.jpg'), img1)
        cv2.imwrite(os.path.join(img_output_dir, file[:-4] + '_2.jpg'), img2)


if __name__ == '__main__':
    img_dir = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/normal'
    json_label_dir = 'D:/pycharm/Medical-images/MedCV/dataset/bp_data/tc_new/guangdong1_round1_train2_20190828/Annotations'
    img_output_dir = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/cutted'
    label_output_dir = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/cutted_data/labels'
    # data_cut(img_dir, json_label_dir, img_output_dir, label_output_dir)
    img_cut(img_dir, img_output_dir)



