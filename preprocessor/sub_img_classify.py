import os


def sub_img_classify_test(label_path):
    category_contain = []
    label_num = 0
    count_category_0, count_category_1, count_category_2 = 0, 0, 0
    with open(label_path, "r") as f:
        labels = f.readlines()
        label_num = len(labels)
        for label in labels:
            if label[0] == "0":
                count_category_0 += 1
            elif label[0] == "1":
                count_category_1 += 1
            elif label[0] == "2":
                count_category_2 += 1

    max_category = "0"
    if count_category_0 > count_category_1:
        if count_category_0 > count_category_2:
            max_category = "0"
        else:
            max_category = "2"
    else:
        if count_category_1 > count_category_2:
            max_category = "1"
        else:
            max_category = "2"

    category_contain.append(max_category)

    # if count_category_0 > 0:
    #     category_contain.append("0")
    # if count_category_1 > 0:
    #     category_contain.append("1")
    # if count_category_2 > 0:
    #     category_contain.append("2")
    return category_contain


def sub_img_classify_true(label_path):
    category_contain = []
    count_category_0, count_category_1, count_category_2 = 0, 0, 0
    with open(label_path, "r") as f:
        labels = f.readlines()
        for label in labels:
            if label[0] == "0":
                count_category_0 += 1
            elif label[0] == "1":
                count_category_1 += 1
            elif label[0] == "2":
                count_category_2 += 1

    if count_category_0 > 0:
        category_contain.append("0")
    if count_category_1 > 0:
        category_contain.append("1")
    if count_category_2 > 0:
        category_contain.append("2")
    return category_contain


def main(true_label_path, test_label_path):
    test_label_files, true_label_files = [], []
    total_sub_img, acc_count, include_count = 0, 0, 0
    for file in os.listdir(test_label_path):
        if file.endswith('txt'):
            test_label_files.append(file)
    for file in os.listdir(true_label_path):
        if file.endswith('txt'):
            true_label_files.append(file)
    sorted(test_label_files)
    sorted(true_label_files)


    # 保证文件名相同
    for true_label_file in true_label_files:
        if true_label_file in test_label_files:
            total_sub_img += 1
            category_true = sub_img_classify_true(true_label_path + "/" + true_label_file)
            category_test = sub_img_classify_test(test_label_path + "/" + true_label_file)
            if category_true == category_test:
                acc_count += 1
            else:
                print([true_label_file, category_true, category_test])
            for category in category_test:
                if category not in category_true:
                    continue
                include_count += 1

    print(str(format(acc_count / total_sub_img * 100, ".2f")) + "%")
    return str(format(acc_count / total_sub_img * 100, ".2f")) + "%"


test_label_path = "./YOLOv6/tools/runs/inference/test"
true_label_path = "./custom_datas_no_overlapped/labels/test"
main(true_label_path, test_label_path)