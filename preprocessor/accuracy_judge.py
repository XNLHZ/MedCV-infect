import math
import os

def sub_img_judge(true_label_file, test_label_file, threshold=5, img_width=640, img_height=640):
    with open(true_label_file, "r") as f:
        true_labels = []
        for line in f.readlines():
            label = line.strip()
            true_labels.append(label)
    with open(test_label_file, "r") as f:
        test_labels = []
        for line in f.readlines():
            label = line.strip()
            test_labels.append(label)

    # 计算检测到的细菌比例
    match_point = 0
    true_label_count = len(true_labels)
    category_mistake, recognize_mistake = 0, 0
    match_point_category_0 = 0
    match_point_category_1 = 0
    match_point_category_2 = 0
    label_count_category_0 = 0
    label_count_category_1 = 0
    label_count_category_2 = 0
    sub_recognize_data = [[match_point_category_0, label_count_category_0],
                          [match_point_category_1, label_count_category_1],
                          [match_point_category_2, label_count_category_2]]

    for true_label in true_labels:
        true_label = true_label.split()
        true_category = true_label[0]
        if true_category == "0":
            label_count_category_0 += 1
        if true_category == "1":
            label_count_category_1 += 1
        if true_category == "2":
            label_count_category_2 += 1

    if true_label_count == 0:
        if len(test_labels) == 0:
            return "100.00%", match_point, true_label_count, category_mistake, sub_recognize_data, recognize_mistake
        else:
            return "0.00%, identify_mistake", match_point, true_label_count, category_mistake, sub_recognize_data, recognize_mistake

    for test_label in test_labels:
        test_label = test_label.split()
        test_category, test_x, test_y = test_label[0], float(test_label[1]) * img_width, float(test_label[2]) * img_height
        min_distance, min_index, match_category = 2000, 0, None
        for index, true_label in enumerate(true_labels):
            true_label = true_label.split()
            true_category, true_x, ture_y = true_label[0], float(true_label[1]) * img_width, float(true_label[2]) * img_width
            temp_distance = math.sqrt(pow(true_x - test_x, 2) + pow(ture_y - test_y, 2))
            if temp_distance < min_distance:
                min_distance = temp_distance
                min_index = index
                match_category = true_category
        if min_distance < threshold:
            if len(true_labels) == 0:
                break
            match_point += 1
            if match_category != test_category:
                category_mistake += 1
            if match_category == "0":
                match_point_category_0 += 1
            elif match_category == "1":
                match_point_category_1 += 1
            elif match_category == "2":
                match_point_category_2 += 1
            del(true_labels[min_index])
        else:
            recognize_mistake += 1

    # 小图识别率
    coverage = format(match_point / true_label_count * 100, ".2f") + "%"

    # 构成大图识别率的数据
    sub_recognize_data = [[match_point_category_0, label_count_category_0],
                          [match_point_category_1, label_count_category_1],
                          [match_point_category_2, label_count_category_2]]
    return coverage, match_point, true_label_count, category_mistake, sub_recognize_data, recognize_mistake


def result_judge(true_label_path, test_label_path, result_file, total_result_file):
    bacteria_coverage = {}
    true_label_files = os.listdir(true_label_path)
    test_label_files = os.listdir(test_label_path)
    total_match_point, total_label_count, category_mistake = 0, 0, 0
    total_match_point_category_0 = 0
    total_match_point_category_1 = 0
    total_match_point_category_2 = 0
    total_label_count_category_0 = 0
    total_label_count_category_1 = 0
    total_label_count_category_2 = 0
    total_recognize_mistake = 0

    for true_label_file in true_label_files:
        if true_label_file in test_label_files:
            data = sub_img_judge(true_label_path + "/" + true_label_file, test_label_path + "/" + true_label_file)
            bacteria_coverage[int(true_label_file[8:-4])] = data[0]
            total_match_point += data[1]
            total_label_count += data[2]
            category_mistake += data[3]
            total_match_point_category_0 += data[4][0][0]
            total_label_count_category_0 += data[4][0][1]
            total_match_point_category_1 += data[4][1][0]
            total_label_count_category_1 += data[4][1][1]
            total_match_point_category_2 += data[4][2][0]
            total_label_count_category_2 += data[4][2][1]
            total_recognize_mistake += data[5]
        else:
            with open(true_label_path + "/" + true_label_file, "r") as f:
                data1 = f.readlines()
                if data1 == []:
                    bacteria_coverage[int(true_label_file[8:-4])] = "100.00%"
                else:
                    bacteria_coverage[int(true_label_file[8:-4])] = "0.00%, fail to identify"
                    total_label_count += len(data1)
                    for label in data1:
                        if label[0] == "0":
                            total_label_count_category_0 += 1
                        if label[0] == "1":
                            total_label_count_category_1 += 1
                        if label[0] == "2":
                            total_label_count_category_2 += 1

    # 生成小图识别率文件
    with open('/'.join(test_label_path.split("/")[:-1]) + "/" + result_file, "w") as f:
        for key in sorted(bacteria_coverage):
            f.write("img_index:" + str(key) + "\tcoverage:" + bacteria_coverage[key] + "\n")

    # 生成总识别率，总准确率，各类细菌识别率的文件
    with open('/'.join(test_label_path.split("/")[:-1]) + "/" + total_result_file, "w") as f:
        f.write("bacteria_recognized_total_coverage:\t\t\t" + format(total_match_point / total_label_count * 100, ".2f")
                + "%\n")
        f.write("·bacteria_recognized_coverage_category_0:\t\t" +
                format(total_match_point_category_0 / total_label_count_category_0 * 100, ".2f") + "%\n")
        f.write("·bacteria_recognized_coverage_category_1:\t\t" +
                format(total_match_point_category_1 / total_label_count_category_1 * 100, ".2f") + "%\n")
        f.write("·bacteria_recognized_coverage_category_2:\t\t" +
                format(total_match_point_category_2 / total_label_count_category_2 * 100, ".2f") + "%\n")
        f.write("bacteria_category_recognized_accuracy:\t\t\t" + format((1.0 - category_mistake / total_match_point) * 100,
                                                                        ".2f") + "%\n")
        f.write("bacteria_mis_recognized_rate:\t\t\t\t" +
                format(total_recognize_mistake / (total_match_point_category_2 + total_recognize_mistake) * 100, ".2f")
                + "%\n")


def main(true_label_path, test_label_path, result_file, total_result_file):
    result_judge(true_label_path, test_label_path, result_file, total_result_file)


if __name__ == "__main__":
    true_label_path = "./custom_datas/labels/test"
    test_label_path = "./YOLOv6/tools/runs/inference/test"
    result_file = "img_bacteria_coverage_test.txt"
    total_result_file = "total_img_bacteria_coverage_test.txt"
    main(true_label_path, test_label_path, result_file, total_result_file)

