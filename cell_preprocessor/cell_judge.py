import copy
import os
import time
import cv2


def check_file(file):
    if not os.path.exists(file):
        os.mknod(file)


def test_cut(img):
    sub_images = []
    height, width = img.shape[:2]
    for h in range(640, 1290, 640):
        for w in range(640, width, 640):
            img1 = copy.deepcopy(img[h-640:h, w-640:w, ...])
            sub_images.append(img1)
            if w+640 > width:
                w = width-1
                img1 = copy.deepcopy(img[h-640:h, w-640:w, ...])
                sub_images.append(img1)
        if h+640 > 1500:
            h = height-1
            for w in range(640, width, 640):
                img1 = copy.deepcopy(img[h - 640:h, w - 640:w, ...])
                sub_images.append(img1)
                if w + 640 > width:
                    w = width - 1
                    img1 = copy.deepcopy(img[h - 640:h, w - 640:w, ...])
                    sub_images.append(img1)
    print(len(sub_images))
    return sub_images

def main(source_image_path, sub_images_path, sub_labels_path):
    # 计时
    # start1 = time.perf_counter()
    # count = 0
    #
    # filt = [0, 5, 12, 17]
    #
    # # 切割图片
    # for file in os.listdir(source_image_path):
    #     img = cv2.imread(os.path.join(source_image_path, file))
    #     sub_images = test_cut(img)
    #     for i, sub_img in enumerate(sub_images):
    #         print(file, i)
    #         if i not in filt:
    #             cv2.imwrite(os.path.join(sub_images_path, file[:-4]+'_'+str(i)+'.jpg'), sub_img)
    #
    # # 计时
    # end1 = time.perf_counter()
    # print("运行时间为", round(end1 - start1), 'seconds')

    # 输入YOLOv6进行检测

    # 对输出进行整理

    # 计时
    start3 = time.perf_counter()

    count_Nm, count_Eg, count_Vc = 0, 0, 0
    for file in os.listdir(sub_labels_path):
        if file.endswith('.txt'):
            with open(os.path.join(sub_labels_path, file), "r") as f:
                labels = f.readlines()
                for label in labels:
                    label = label.strip()
                    label = label.split(' ')
                    if label[6] == "Nm":
                        count_Nm += 1
                    elif label[6] == "Eg":
                        count_Eg += 1
                    elif label[6] == "Vc":
                        count_Vc += 1


    # 计算比例
    total_count = count_Nm + count_Eg + count_Vc

    print('普通细胞计数：', count_Nm)
    print('吞噬细胞计数：', count_Eg)
    print('空泡细胞计数：', count_Vc)
    print('总细胞计数：', total_count)
    print('包含吞噬与空泡的细胞百分率：', '{:.2f}'.format((count_Eg + count_Vc) / total_count * 100)+'%')

    # 计时
    end3 = time.perf_counter()
    print("运行时间为", round(end3 - start3), 'seconds')


if __name__ == "__main__":
    source_image_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/3_part_val/original_data/6.16-PA'
    sub_images_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/3_part_val/crop_data/6.16-PA'
    sub_labels_path = 'D:/pycharm/Medical-images/MedCV/YOLOv6/cell_data/3_part_val/output/6.16-PA/6.16-PA'
    # main(source_image_path, sub_images_path, sub_labels_path)

    p = 0.7167
    r = 0.8817
    f1 = 2*p*r / (p+r)
    print(f1)