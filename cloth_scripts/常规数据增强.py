import cv2
import os
import random
import copy
from PIL import ImageEnhance
from PIL import Image
import numpy as np


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# 高斯噪音
def gauss_noise(img, labels):
    img_height, img_width, img_channels = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
    noisy_img = img + gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img, labels


# 泊松噪声
def bs_noise(img, labels):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img, labels


# 随机调整图像的饱和度和亮度
def randomColor(img, labels):
    img = img.astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 随机生成0，1来随机确定调整哪个参数，可能会调整饱和度，也可能会调整图像的饱和度和亮度
    saturation = random.randint(0, 1)
    brightness = random.randint(0, 1)
    contrast = random.randint(0, 1)
    sharpness = random.randint(0, 1)

    # 当三个参数中一个参数为1，就可执行相应的操作
    if random.random() < saturation:
        random_factor = np.random.randint(90, 110) / 100.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(90, 110) / 100.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(90, 110) / 100.  # 随机因子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(90, 110) / 100.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img, labels


# 旋转
def random_rect_rotate(img, labels, angle):
    height, width = img.shape[:2]
    new_img = None
    new_labels = copy.deepcopy(labels)
    if angle == 0:
        return img, labels
    elif angle == 270:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[height - j - 1][i] = img[i][j].copy()
        for data in new_labels:
            data[1], data[2] = data[2], str(format(1.0 - float(data[1]), ".6f"))
            data[3], data[4] = data[4], data[3]

    elif angle == 180:
        new_img = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[height - i - 1][width - j - 1] = img[i][j].copy()
        for data in new_labels:
            data[1], data[2] = str(format(1.0 - float(data[1]), ".6f")), str(format(1.0 - float(data[2]), ".6f"))

    elif angle == 90:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[j][width - i - 1] = img[i][j].copy()
        for data in new_labels:
            data[1], data[2] = str(format(1.0 - float(data[2]), ".6f")), data[1]
            data[3], data[4] = data[4], data[3]

    return new_img, new_labels


# 镜像
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
            data[2] = str(format(1.0 - float(data[2]), ".6f"))

    # 45°镜像
    if angle == 45:
        for i in range(height):
            for j in range(width-i):
                new_img[i][j] = img[width-j-1][width-i-1].copy()
                new_img[width-j-1][width-i-1] = img[i][j].copy()
        for data in new_labels:
            data[1], data[2] = str(format(1.0 - float(data[2]), ".6f")), str(format(1.0 - float(data[1]), ".6f"))

    # 90°镜像
    if angle == 90:
        for i in range(height):
            for j in range(int(width/2)):
                new_img[i][j] = img[i][width-j-1].copy()
                new_img[i][width-j-1] = img[i][j].copy()
        for data in new_labels:
            data[1] = str(format(1.0 - float(data[1]), ".6f"))

    # 135°镜像
    if angle == 135:
        for i in range(height):
            for j in range(i):
                new_img[i][j] = img[j][i].copy()
                new_img[j][i] = img[i][j].copy()
        for data in new_labels:
            data[1], data[2] = str(format(float(data[2]), ".6f")), str(format(float(data[1]), ".6f"))

    return new_img, new_labels


# 图像融合
def img_overlap(img1, label1, img2, label2):
    label = label1 + label2
    height, width = img1.shape[:2]
    for i in range(height):
        for j in range(width):
            img1[i][j][0] = min(img1[i][j][0], img2[i][j][0])
            img1[i][j][1] = min(img1[i][j][1], img2[i][j][1])
            img1[i][j][2] = min(img1[i][j][2], img2[i][j][2])
    return img1, label


def test():
    image_path = 'D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetA/train/images/sub_img_0.png'
    label_path = 'D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetA/train/labels/sub_img_0.txt'
    out_path = 'D:/pycharm/Medical-images/MedCV/test'
    out_mark_path = 'D:/pycharm/Medical-images/MedCV/test'
    img = cv2.imread(image_path)
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')
            labels.append(line)

    img1, label1 = copy.deepcopy(gauss_noise(img, labels))
    cv2.imwrite(os.path.join(out_path, 'gauss_noise.jpg'), img1)
    with open(os.path.join(out_path, 'gauss_noise.txt'), 'w') as f:
        for i, label in enumerate(label1):
            label1[i] = ' '.join(label)+'\n'
        f.writelines(label1)
    point_mark(img1, label1, 'gauss_noise', out_mark_path)

    img2, label2 = copy.deepcopy(bs_noise(img, labels))
    cv2.imwrite(os.path.join(out_path, 'bs_noise.jpg'), img2)
    with open(os.path.join(out_path, 'bs_noise.txt'), 'w') as f:
        for i, label in enumerate(label2):
            label2[i] = ' '.join(label)+'\n'
        f.writelines(label2)
    point_mark(img2, label2, 'bs_noise', out_mark_path)

    img3, label3 = copy.deepcopy(randomColor(img, labels))
    cv2.imwrite(os.path.join(out_path, 'randomColor.jpg'), img3)
    with open(os.path.join(out_path, 'randomColor.txt'), 'w') as f:
        for i, label in enumerate(label3):
            label3[i] = ' '.join(label)+'\n'
        f.writelines(label3)
    point_mark(img3, label3, 'randomColor', out_mark_path)

    img4, label4 = copy.deepcopy(random_rect_rotate(img, labels, 90))
    cv2.imwrite(os.path.join(out_path, 'rotate_90.jpg'), img4)
    with open(os.path.join(out_path, 'rotate_90.txt'), 'w') as f:
        for i, label in enumerate(label4):
            label4[i] = ' '.join(label)+'\n'
        f.writelines(label4)
    point_mark(img4, label4, 'rotate_90', out_mark_path)

    img5, label5 = copy.deepcopy(random_rect_rotate(img, labels, 180))
    cv2.imwrite(os.path.join(out_path, 'rotate_180.jpg'), img5)
    with open(os.path.join(out_path, 'rotate_180.txt'), 'w') as f:
        for i, label in enumerate(label5):
            label5[i] = ' '.join(label)+'\n'
        f.writelines(label5)
    point_mark(img5, label5, 'rotate_180', out_mark_path)

    img6, label6 = copy.deepcopy(random_rect_rotate(img, labels, 270))
    cv2.imwrite(os.path.join(out_path, 'rotate_270.jpg'), img6)
    with open(os.path.join(out_path, 'rotate_270.txt'), 'w') as f:
        for i, label in enumerate(label6):
            label6[i] = ' '.join(label)+'\n'
        f.writelines(label6)
    point_mark(img6, label6, 'rotate_270', out_mark_path)

    img7, label7 = copy.deepcopy(mirror(img, labels, 0))
    cv2.imwrite(os.path.join(out_path, 'mirror_0.jpg'), img7)
    with open(os.path.join(out_path, 'mirror_0.txt'), 'w') as f:
        for i, label in enumerate(label7):
            label7[i] = ' '.join(label)+'\n'
        f.writelines(label7)
    point_mark(img7, label7, 'mirror_0', out_mark_path)

    img8, label8 = copy.deepcopy(mirror(img, labels, 45))
    cv2.imwrite(os.path.join(out_path, 'mirror_45.jpg'), img8)
    with open(os.path.join(out_path, 'mirror_45.txt'), 'w') as f:
        for i, label in enumerate(label8):
            label8[i] = ' '.join(label)+'\n'
        f.writelines(label8)
    point_mark(img8, label8, 'mirror_45', out_mark_path)

    img9, label9 = copy.deepcopy(mirror(img, labels, 90))
    cv2.imwrite(os.path.join(out_path, 'mirror_90.jpg'), img9)
    with open(os.path.join(out_path, 'mirror_90.txt'), 'w') as f:
        for i, label in enumerate(label9):
            label9[i] = ' '.join(label)+'\n'
        f.writelines(label9)
    point_mark(img9, label9, 'mirror_90', out_mark_path)

    img10, label10 = copy.deepcopy(mirror(img, labels, 135))
    cv2.imwrite(os.path.join(out_path, 'mirror_135.jpg'), img10)
    with open(os.path.join(out_path, 'mirror_135.txt'), 'w') as f:
        for i, label in enumerate(label10):
            label10[i] = ' '.join(label)+'\n'
        f.writelines(label10)
    point_mark(img10, label10, 'mirror_135', out_mark_path)


def random_augument(image_dir, label_dir, out_img_file, out_label_file, target_count):
    # 设定各数据增强方法的触发概率
    pbb_gauss_noise = 0.1
    pbb_bs_noise = 0.1
    pbb_random_Color = 0.05
    pbb_random_rotate = 0.0
    pbb_random_mirror = 0.0
    pbb_image_merge = 0
    image_merge_flag = 0
    pbb_same_class_merge = 0.8

    images = []
    labels = []
    img_len = len(os.listdir(image_dir))
    for image in os.listdir(image_dir):
        images.append(cv2.imread(os.path.join(image_dir, image)))
        with open(os.path.join(label_dir, image[:-4]+'.txt'), 'r') as f:
            label = []
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                label.append(line)
            labels.append(label)
    count = 0
    print(len(images), len(labels))
    while count < target_count:
        # 选择初始图片
        idx = random.randint(1, img_len)
        if image_merge_flag == 1:
            idx2 = random.randint(1, img_len)
            if random.random() < pbb_same_class_merge:
                while labels[idx2 - 1] == [] or labels[idx - 1] == [] or labels[idx2 - 1][0][0] != labels[idx - 1][0][0]:
                    if labels[idx - 1] == []:
                        idx = random.randint(1, img_len)
                    else:
                        idx2 = random.randint(1, img_len)
            img2, label2 = copy.deepcopy(images[idx2 - 1]), copy.deepcopy(labels[idx2 - 1])
        img, label = copy.deepcopy(images[idx-1]), copy.deepcopy(labels[idx-1])
        # 判定是否进行各数据增强操作
        if random.random() < pbb_gauss_noise:
            img, label = gauss_noise(img, label)
            if image_merge_flag == 1:
                img2, label2 = gauss_noise(img2, label2)
        elif random.random() < pbb_bs_noise:
            img, label = bs_noise(img, label)
            if image_merge_flag == 1:
                img2, label2 = bs_noise(img2, label2)
        if random.random() < pbb_random_Color:
            img, label = randomColor(img, label)
            if image_merge_flag == 1:
                img2, label2 = randomColor(img2, label2)
        if random.random() < pbb_random_rotate:
            angle = random.randint(0, 3) * 90
            img, label = random_rect_rotate(img, label, angle)
            if image_merge_flag == 1:
                img2, label2 = random_rect_rotate(img2, label2, angle)
        if random.random() < pbb_random_mirror:
            angle = random.randint(0, 3) * 45
            img, label = mirror(img, label, angle)
            if image_merge_flag == 1:
                img2, label2 = mirror(img2, label2, angle)

        # 图像融合法
        if image_merge_flag == 1:
            if random.random() < pbb_image_merge:
                img, label = img_overlap(img, label, img2, label2)

        # 保存新生成的图像
        cv2.imwrite(os.path.join(out_img_file, 'normal_augument_' + str(count) + '.jpg'), img)
        with open(os.path.join(out_label_file, 'normal_augument_' + str(count) + '.txt'), 'w') as f:
            for i, lb in enumerate(label):
                label[i] = ' '.join(lb) + '\n'
            f.writelines(label)
        count += 1


def augument(image_dir, label_dir, out_img_file, out_label_file):
    count = 0
    images = []
    labels = []
    img_len = len(os.listdir(image_dir))
    for image in os.listdir(image_dir):
        images.append(cv2.imread(os.path.join(image_dir, image)))
        with open(os.path.join(label_dir, image[:-4]+'.txt'), 'r') as f:
            label = []
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                label.append(line)
            labels.append(label)
    for i, img in enumerate(images):
        for angle in range(0, 360, 90):
            new_img1, new_label1 = copy.deepcopy(random_rect_rotate(img, labels[i], angle))
            if angle < 180:
                for angle1 in range(0, 270, 90):
                    if angle1 != 180:
                        new_img2, new_label2 = copy.deepcopy(mirror(new_img1, new_label1, angle1))
                    else:
                        new_img2, new_label2 = copy.deepcopy(new_img1), new_label1
                    cv2.imwrite(os.path.join(out_img_file, 'augument_' + str(count) + '_' + str(angle) + '_' + str(angle1) + '.jpg'), new_img2)
                    with open(os.path.join(out_label_file, 'augument_' + str(count) + '_' + str(angle) + '_' + str(angle1) + '.txt'), 'w') as f:
                        for k, lb in enumerate(new_label2):
                            new_label2[k] = ' '.join(lb) + '\n'
                        f.writelines(new_label2)
            else:
                angle1 = 0
                cv2.imwrite(os.path.join(out_img_file, 'augument_' + str(count) + '_' + str(angle) + '_' + str(angle1) + '.jpg'), new_img1)
                with open(os.path.join(out_label_file, 'augument_' + str(count) + '_' + str(angle) + '_' + str(angle1) + '.txt'), 'w') as f:
                    for j, lb in enumerate(new_label1):
                        new_label1[j] = ' '.join(lb) + '\n'
                    f.writelines(new_label1)
        count += 1



# 标注点可视化
def point_mark(img, labels, name='', out_file="mark_images"):
    color = (0, 0, 255)
    height, width = img.shape[:2]
    labels1 = copy.deepcopy(labels)
    # for i, label in enumerate(labels1):
    #     labels1[i] = label.split(' ')
    for data in labels1:
        x, y = int(float(data[1]) * width), int(float(data[2]) * height)
        cv2.circle(img, (x, y), 5, color, 1)
    cv2.imwrite(out_file + "/" + name + "_mark.jpg", img)


if __name__ == "__main__":
    # image_dir: 存储原图片数据的路径
    # label_dir: 存储原YOLO标注数据的路径
    # out_img_file: 输出增强后图片的路径
    # out_label_file: 增强后yolo格式标注数据的存储路径

    image_dir = "D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/divide_data/cutted/images/test"
    label_dir = "D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/divide_data/cutted/labels/test"
    out_img_file = "D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/augument_data/cutted/images/test"
    out_label_file = "D:/pycharm/Medical-images/MedCV/dataset/cloth_YOLO_data/augument_data/cutted/labels/test"
    # target_count = 20000

    augument(image_dir, label_dir, out_img_file, out_label_file)
    # random_augument(image_dir, label_dir, out_img_file, out_label_file, target_count)
    # test()

    # 标注可视化
    # image_path = 'D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetC/50000/train/images1/augument_1648_90_90.jpg'
    # label_path = 'D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetC/50000/train/labels1/augument_1648_90_90.txt'
    # out_file = 'D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetC/50000/train'
    # img = cv2.imread(image_path)
    # label = []
    # with open(label_path, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')
    #         line = line.split(' ')
    #         label.append(line)
    # point_mark(img, label, '1', out_file)