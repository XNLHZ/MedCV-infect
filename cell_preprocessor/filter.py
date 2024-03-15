import os


def filter(image_path, label_path):
    for file in os.listdir(label_path):
        flag = 0
        with open(os.path.join(label_path, file), 'r') as f:
            if f.read() == '':
                flag = 1
        if flag == 1:
            os.remove(os.path.join(label_path, file))
            os.remove(os.path.join(image_path, file[:-4] + '.png'))




def main():
    image_path = 'D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/images/test'
    label_path = 'D:/pycharm/Medical-images/MedCV/cell_6_classify_prepare/labels/test'
    filter(image_path, label_path)



main()