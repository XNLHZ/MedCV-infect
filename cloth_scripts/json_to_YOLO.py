import os
import json


def json_to_YOLO(json_dir, output_dir):
    for json_file in os.listdir(json_dir):
        with open(os.path.join(json_dir, json_file), "r", encoding="utf8") as fp:
            json_datas = json.load(fp)
            height, width = 1000, 2446
            yolo_txt = []
            for label in json_datas:
                xmin = int(label['bbox'][0])
                ymin = int(label['bbox'][1])
                xmax = int(label['bbox'][2])
                ymax = int(label['bbox'][3])
                yolo_label = ['0', str(((xmax - xmin) / 2.0 + xmin) / width),
                         str(((ymax - ymin) / 2.0 + ymin) / height), str((ymax - ymin) / width),
                         str((ymax - ymin) / height)]
                str_label = ' '.join(yolo_label)
                yolo_txt.append(yolo_label)
                print(str_label)
                with open(os.path.join(output_dir, label['name'][:-3] + 'txt'), 'w') as f:
                    f.write(str_label)


if __name__ == '__main__':
    json_dir = 'D:/pycharm/Medical-images/MedCV/dataset/布匹照片/天池new/guangdong1_round1_train2_20190828/Annotations'
    output_dir = 'D:/pycharm/Medical-images/MedCV/dataset/布匹照片/天池new/guangdong1_round1_train2_20190828/YOLO_label'
    json_to_YOLO(json_dir, output_dir)