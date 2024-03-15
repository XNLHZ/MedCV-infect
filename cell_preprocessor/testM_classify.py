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
    for h in range(640, height, 640):
        for w in range(640, width, 640):
            img1 = copy.deepcopy(img[h-640:h, w-640:w, ...])
            sub_images.append(img1)
            if w+640 > width:
                w = width-1
                img1 = copy.deepcopy(img[h-640:h, w-640:w, ...])
                sub_images.append(img1)
        if h+640 > height:
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
    #
    dic, count = {}, 0

    # 生成标签与文件名的对应字典
    dic['AB'], dic['KP'], dic['PA'] = [], [], []
    #
    # 切割图片
    for file in os.listdir(os.path.join(source_image_path, 'AB')):
        dic['AB'].append(file)
    #     img = cv2.imread(os.path.join(source_image_path, 'AB', file))
    #     sub_images = test_cut(img)
    #     for i, sub_img in enumerate(sub_images):
    #         print(file, i)
    #         cv2.imwrite(os.path.join(sub_images_path, file[:-4]+'_'+str(i)+'.jpg'), sub_img)
    #
    for file in os.listdir(os.path.join(source_image_path, 'KP')):
        dic['KP'].append(file)
    #     img = cv2.imread(os.path.join(source_image_path, 'KP', file))
    #     sub_images = test_cut(img)
    #     for i, sub_img in enumerate(sub_images):
    #         print(file, i)
    #         cv2.imwrite(os.path.join(sub_images_path, file[:-4]+'_'+str(i)+'.jpg'), sub_img)
    #
    for file in os.listdir(os.path.join(source_image_path, 'PA')):
        dic['PA'].append(file)
    #     img = cv2.imread(os.path.join(source_image_path, 'PA', file))
    #     sub_images = test_cut(img)
    #     for i, sub_img in enumerate(sub_images):
    #         print(file, i)
    #         cv2.imwrite(os.path.join(sub_images_path, file[:-4]+'_'+str(i)+'.jpg'), sub_img)
    #
    # # 计时
    # end1 = time.perf_counter()
    # print("运行时间为", round(end1 - start1), 'seconds')

    # 输入YOLOv6进行检测

    # 对输出进行整理

    # 计时
    start3 = time.perf_counter()

    image_judges = []
    count_AB, count_KP, count_PA = 0, 0, 0
    for file in os.listdir(sub_labels_path):
        if file.endswith('.jpg'):
            check_file(os.path.join(sub_labels_path, file[:-4]+'.txt'))
    for file in os.listdir(sub_labels_path):
        if file.endswith('.txt'):
            with open(os.path.join(sub_labels_path, file), "r") as f:
                labels = f.readlines()
                for label in labels:
                    if label[0] == "0":
                        count_AB += 1
                    elif label[0] == "1":
                        count_KP += 1
                    elif label[0] == "2":
                        count_PA += 1
                if file.split('_')[1].split('.')[0] == '9':
                    print([file, [count_AB, count_KP, count_PA]])
                    total = count_AB + count_KP + count_PA
                    image_judges.append([file.split(')')[0] + ').jpg', [count_AB > total * 0.75, count_KP > total * 0.75, count_PA > total * 0.75]])



    # 计算ACC
    ac_count, total_count = 0, 0
    for judge in image_judges:
        total_count += 1
        if judge[0] in dic['AB']:
            if judge[1] == [1, 0, 0]:
                ac_count += 1
            else:
                print(judge[0])
        elif judge[0] in dic['KP']:
            if judge[1] == [0, 1, 0]:
                ac_count += 1
            else:
                print(judge[0])
        else:
            if judge[1] == [0, 0, 1]:
                ac_count += 1
            else:
                print(judge[0])

    ACC = format(ac_count / total_count, '.6f')
    print(ac_count, total_count)
    print(ACC)

    # 计时
    end3 = time.perf_counter()
    print("运行时间为", round(end3 - start3), 'seconds')


if __name__ == "__main__":
    source_image_path = '/home/zccao20215227103/hcxue/MedCV/dataset/AB_KP_PA'
    sub_images_path = '/home/zccao20215227103/hcxue/MedCV/dataset/AB_KP_PA/sub_images'
    sub_labels_path = '/home/zccao20215227103/hcxue/MedCV/YOLOv6/runs/testM_03_02/sub_images'
    main(source_image_path, sub_images_path, sub_labels_path)
