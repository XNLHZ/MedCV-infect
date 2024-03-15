import os
import cv2


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def output_trans(image_path, output_label_path, out_path):
    check_dir(os.path.join(out_path, '0'))
    check_dir(os.path.join(out_path, '1'))
    check_dir(os.path.join(out_path, '2'))
    for image in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image))
        height, width = img.shape[:2]
        with open(os.path.join(output_label_path, image[-4]+'.txt'), 'r') as f:
            count0, count1, count2 = 0, 0, 0
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                left = line[1] * width - line[3] * width / 2
                right = line[1] * width + line[3] * width / 2
                up = line[2] * height - line[4] * height / 2
                down = line[2] * height + line[4] * height / 2
                sub_img = img[up:down, left:right, ...]
                if line[0] == '0':
                    cv2.imwirte(os.path.join(out_path, '0', image[:-4] + '_' + str(count0) + '.jpg'), sub_img)
                    count0 += 1
                elif line[0] == '1':
                    cv2.imwirte(os.path.join(out_path, '1', image[:-4] + '_' + str(count1) + '.jpg'), sub_img)
                    count1 += 1
                else:
                    cv2.imwirte(os.path.join(out_path, '2', image[:-4] + '_' + str(count2) + '.jpg'), sub_img)
                    count2 += 1




if __init__ == '__main__':
    image_path = ''
    output_label_path = ''
    out_path = ''
    output_trans(image_path, output_label_path, out_path)
    