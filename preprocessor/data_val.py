import cv2


def point_mark(img, label_data, index="", out_file="mark_images"):
    color = (0, 0, 255)
    height, width = img.shape[:2]
    for data in label_data:
        x, y = int(float(data[1]) * height), int(float(data[2]) * width)
        cv2.circle(img, (x, y), 50, color, 1)
    cv2.imwrite(out_file + "/" + "mark_img_" + index + ".png", img)


def cut_img_mark(img, label_file, index, out_file):
    label_data = []
    with open(label_file, "r", encoding="utf8") as f:
        line = f.readline()
        line = line[:-1]
        while line:
            label_data.append(line.split())
            line = f.readline()
            line = line[:-1]
        print(label_data)
    point_mark(img, label_data, str(index), out_file)


def data_val(img_path, label_path, num, out_file):
    # for i in range(num):
    #     img = cv2.imread(img_path + "/sub_img_" + str(i) + ".png")
    #     cut_img_mark(img, label_path + "/sub_img_" + str(i) + ".txt", i, out_file)

    for i in range(num):
        for j in range(4):
            img = cv2.imread(img_path + "/sub_img_" + str(i+100) + "_" + str(j * 45) + ".png")
            cut_img_mark(img, label_path + "/sub_img_" + str(i+100) + "_" + str(j * 45) + ".txt", str(i) + "_" + str(j * 45), out_file)


image_path = "./custom_datas/images/train"        # 想要画出的图片路径
label_path = "./custom_datas/labels/train"        # 想要画出的标注路径
out_file = "./custom_datas/mark_images"     # 圈出标注点后输出图片的路径
data_val(image_path, label_path, 10, out_file)
