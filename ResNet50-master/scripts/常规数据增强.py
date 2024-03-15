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
def gauss_noise(img):
    img_height, img_width, img_channels = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
    noisy_img = img + gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img


# 泊松噪声
def bs_noise(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img


# 随机调整图像的饱和度和亮度
def randomColor(img):
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
    return img


# 旋转
def random_rect_rotate(img, angle):
    height, width = img.shape[:2]
    new_img = None
    # new_labels = copy.deepcopy(labels)
    if angle == 0:
        return img
    elif angle == 270:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[height - j - 1][i] = img[i][j]
        # for data in new_labels:
        #     data[1], data[2] = data[2], str(format(1.0 - float(data[1]), ".6f"))
        #     data[3], data[4] = data[4], data[3]

    elif angle == 180:
        new_img = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[height - i - 1][width - j - 1] = img[i][j]
        # for data in new_labels:
        #     data[1], data[2] = str(format(1.0 - float(data[1]), ".6f")), str(format(1.0 - float(data[2]), ".6f"))

    elif angle == 90:
        new_img = np.zeros((width, height, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                new_img[j][width - i - 1] = img[i][j]
        # for data in new_labels:
        #     data[1], data[2] = str(format(1.0 - float(data[2]), ".6f")), data[1]
        #     data[3], data[4] = data[4], data[3]

    return new_img


# 镜像
def mirror(img, angle):
    # 由于yolo输入数据为640*640，可以支持45°步长的镜像
    height, width = img.shape[:2]
    new_img = copy.deepcopy(img)
    # new_labels = copy.deepcopy(labels)

    # 0°镜像
    if angle == 0:
        for i in range(int(height/2)):
            for j in range(width):
                new_img[i][j] = img[height-i-1][j].copy()
                new_img[height-i-1][j] = img[i][j].copy()
        # for data in new_labels:
        #     data[2] = str(format(1.0 - float(data[2]), ".6f"))

    # 45°镜像
    if angle == 45:
        for i in range(height):
            for j in range(width-i):
                new_img[i][j] = img[width-j-1][width-i-1].copy()
                new_img[width-j-1][width-i-1] = img[i][j].copy()
        # for data in new_labels:
        #     data[1], data[2] = str(format(1.0 - float(data[2]), ".6f")), str(format(1.0 - float(data[1]), ".6f"))

    # 90°镜像
    if angle == 90:
        for i in range(height):
            for j in range(int(width/2)):
                new_img[i][j] = img[i][width-j-1].copy()
                new_img[i][width-j-1] = img[i][j].copy()
        # for data in new_labels:
        #     data[1] = str(format(1.0 - float(data[1]), ".6f"))

    # 135°镜像
    if angle == 135:
        for i in range(height):
            for j in range(i):
                new_img[i][j] = img[j][i].copy()
                new_img[j][i] = img[i][j].copy()
        # for data in new_labels:
        #     data[1], data[2] = str(format(float(data[2]), ".6f")), str(format(float(data[1]), ".6f"))

    return new_img


# 图像融合
def img_overlap(img1, img2):
    height, width = img1.shape[:2]
    for i in range(height):
        for j in range(width):
            img1[i][j][0] = min(img1[i][j][0], img2[i][j][0])
            img1[i][j][1] = min(img1[i][j][1], img2[i][j][1])
            img1[i][j][2] = min(img1[i][j][2], img2[i][j][2])
    return img1


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


def augument(image_dir, out_img_file, target_count):
    # 设定各数据增强方法的触发概率
    pbb_gauss_noise = 0.0
    pbb_bs_noise = 0.0
    pbb_random_Color = 0.0
    pbb_random_rotate = 0.3
    pbb_random_mirror = 0.4
    pbb_image_merge = 0.0
    image_merge_flag = 0

    images = []
    # labels = []
    img_len = len(os.listdir(image_dir))
    for image in os.listdir(image_dir):
        images.append(cv2.imread(os.path.join(image_dir, image)))
        # with open(os.path.join(label_dir, image[:-4]+'.txt'), 'r') as f:
        #     label = []
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         line = line.split(' ')
        #         label.append(line)
        #     labels.append(label)
    count = 0
    while count < target_count:
        # 选择初始图片
        idx = random.randint(1, img_len)
        if image_merge_flag == 1:
            idx2 = random.randint(1, img_len)
            img2 = copy.deepcopy(images[idx2 - 1])
            img2 = Square_format(img2)
        img = copy.deepcopy(images[idx-1])
        img = Square_format(img)
        # 判定是否进行各数据增强操作
        if random.random() < pbb_gauss_noise:
            img = gauss_noise(img)
            if image_merge_flag == 1:
                img2 = gauss_noise(img2)
        if random.random() < pbb_bs_noise:
            img = bs_noise(img)
            if image_merge_flag == 1:
                img2 = bs_noise(img2)
        if random.random() < pbb_random_Color:
            img = randomColor(img)
            if image_merge_flag == 1:
                img2 = randomColor(img2)
        if random.random() < pbb_random_rotate:
            angle = random.randint(0, 3) * 90
            img = random_rect_rotate(img, angle)
            if image_merge_flag == 1:
                img2 = random_rect_rotate(img2, angle)
        if random.random() < pbb_random_mirror:
            angle = random.randint(0, 3) * 45
            img = mirror(img, angle)
            if image_merge_flag == 1:
                img2 = mirror(img2, angle)

        # 图像融合法
        # if image_merge_flag == 1:
        #     if random.random() < pbb_image_merge:
        #         img, label = img_overlap(img, label, img2, label2)

        # 保存新生成的图像
        cv2.imwrite(os.path.join(out_img_file, 'normal_augument_' + str(count) + '.jpg'), img)
        # with open(os.path.join(out_label_file, 'normal_augument_' + str(count) + '.txt'), 'w') as f:
        #     for i, lb in enumerate(label):
        #         label[i] = ' '.join(lb) + '\n'
        #     f.writelines(label)
        count += 1


# 标注点可视化
def point_mark(img, labels, name='', out_file="mark_images"):
    color = (0, 0, 255)
    height, width = img.shape[:2]
    labels1 = copy.deepcopy(labels)
    for i, label in enumerate(labels1):
        labels1[i] = label.split(' ')
    for data in labels1:
        x, y = int(float(data[1]) * width), int(float(data[2]) * height)
        cv2.circle(img, (x, y), 5, color, 1)
    cv2.imwrite(out_file + "/" + name + "_mark.jpg", img)


# 填充图像为正方形，而且要能保证填充后的图像在0到360°旋转的时候，原图像的像素不会损失
def Square_format(img):
    rows, cols = img.shape[:2]
    if rows > cols:
        re = cv2.copyMakeBorder(img, 0, 0, int((rows - cols) / 2), int(rows - cols - int((rows - cols) / 2)),
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    elif rows < cols:
        re = cv2.copyMakeBorder(img, int((cols - rows) / 2), int(cols - rows - int((cols - rows) / 2)), 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        return img
    return re


if __name__ == "__main__":
    # image_dir: 存储原图片数据的路径
    # label_dir: 存储原YOLO标注数据的路径
    # out_img_file: 输出增强后图片的路径
    # out_label_file: 增强后yolo格式标注数据的存储路径


    image_dir = "D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/test/original_data_nm"
    # label_dir = "D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetC/50000/train/labels"
    out_img_file = "D:/pycharm/Medical-images/MedCV/ResNet50-master/cell_data/test/Square_test_nm"
    # out_label_file = "D:/pycharm/Medical-images/MedCV/dataset/paper_datasets/datasetC/50000/train/augument_labels"
    target_count = 2500

    # augument(image_dir, out_img_file, target_count)
    # test()
    for image in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, image))
        img = Square_format(img)
        cv2.imwrite(out_img_file + '/' + image, img)
