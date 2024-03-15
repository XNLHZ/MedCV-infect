import os
from xml.dom.minidom import parse
import shutil

dir = {'擦洞': '0', '吊经': '1', '毛斑': '2', '毛洞': '3', '织稀': '4', '缺经': '5', '跳花': '6', '扎洞': '7', '污渍': '8',
       '边缺经': '5', '经跳花': '6', '边扎洞': '7', '黄渍': '8', '油渍': '8',
       '边白印': '9', '边缺纬': '9', '边针眼': '9', '擦毛': '9', '擦伤': '9',
       '粗纱': '9', '吊弓': '9', '耳朵': '9', '弓纱': '9', '厚薄段': '9',
       '厚段': '9', '回边': '9', '剪洞': '9', '结洞': '9', '紧纱': '9',
       '经粗纱': '9', '楞断': '9', '愣断': '9',  '毛粒': '9', '破边': '9',
       '破洞': '9', '嵌结': '9', '缺纬': '9', '纬粗纱': '9', '线印': '9',
       '修印': '9', '扎纱': '9', '扎梳': '9', '蒸呢印': '9', '织入': '9',
       '夹码': '9', '明嵌线': '9', '吊纬': '9'}

count = [0 for _ in range(10)]

def toYOLO(file_path, label_out_path, img_out_path):
    for file in os.listdir(file_path):
        if file == '正常':
            continue
            # for file1 in os.listdir(os.path.join(file_path, file)):
            #     if file1.endswith('jpg'):
            #         shutil.copyfile(os.path.join(file_path, file, file1), os.path.join(img_out_path, file1))
            #     with open(os.path.join(label_out_path, file1[:-3] + 'txt'), 'w') as f:
            #         f.write('')
        else:
            for file1 in os.listdir(os.path.join(file_path, file)):
                if file1.endswith('jpg'):
                    shutil.copyfile(os.path.join(file_path, file, file1), os.path.join(img_out_path, file1))
                if file1.endswith('xml'):
                    domtree = parse(os.path.join(file_path, file, file1))  # 解析xml文件
                    root_node = domtree.documentElement  # .documentElement：获取根节点
                    size = root_node.getElementsByTagName("size")[0]
                    width = int(size.getElementsByTagName("width")[0].childNodes[0].data)
                    height = int(size.getElementsByTagName("height")[0].childNodes[0].data)

                    object = root_node.getElementsByTagName("object")[0]
                    bndbox = object.getElementsByTagName("bndbox")[0]
                    xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
                    xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
                    ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
                    ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)

                    label = [dir[file], str(((xmax - xmin) / 2.0 + xmin) / width),
                             str(((ymax - ymin) / 2.0 + ymin) / height), str((ymax - ymin) / width),
                             str((ymax - ymin) / height)]
                    count[int(dir[file])] += 1
                    str_label = ' '.join(label)
                    print(str_label)
                    with open(os.path.join(label_out_path, file1[:-3]+'txt'), 'w') as f:
                        f.write(str_label)
                    print(count)




if __name__ ==  '__main__':
    file_path1 = 'C:/布匹照片/天池/初赛1/xuelang_round1_train_part1_20180628'
    file_path2 = 'C:/布匹照片/天池/初赛1/xuelang_round1_train_part2_20180705'
    file_path3 = 'C:/布匹照片/天池/初赛1/xuelang_round1_train_part3_20180709'
    train_label_out_path = 'C:/布匹照片/天池/初赛1/cloth_YOLO_data/label'
    train_img_out_path = 'C:/布匹照片/天池/初赛1/cloth_YOLO_data/img'

    # 训练集数据转换
    toYOLO(file_path2, train_label_out_path, train_img_out_path)
    toYOLO(file_path3, train_label_out_path, train_img_out_path)
    toYOLO(file_path1, train_label_out_path, train_img_out_path)


