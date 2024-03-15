import os


def two_class_judge(true_label_path, infer_label_path, image_path):
    count_total, count_true = 0, 0
    for file in os.listdir(image_path):
        if file.endswith('jpg'):
            flag1, flag2 = 0, 0
            if os.path.exists(os.path.join(true_label_path, file[:-3]+'txt')):
                flag1 = 1
            if os.path.exists(os.path.join(infer_label_path, file[:-3] + 'txt')):
                flag2 = 1
            count_total += 1
            if flag1 == flag2:
                count_true += 1

    acc = format(count_true / count_total, '.6f')
    print(count_total, count_true, acc)


if __name__ == '__main__':
    true_label_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/images'
    infer_label_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6-main/runs/cloth_infer/infer_cutted_test_0.35/labels'
    image_path = 'D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/normal_test/cutted/images'
    two_class_judge(true_label_path, infer_label_path, image_path)

