import os
import copy


def data_balance(label_path, new_label_path):
    for file in os.listdir(label_path):
        labels = []
        with open(os.path.join(label_path, file), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                labels.append(line)
        for label in labels:
            if label[0] == '0' or label[0] == '1':
                new_label1, new_label2 = copy.deepcopy(label), copy.deepcopy(label)
                new_label1[1] = str(max(float(new_label1[1])-0.003125, 0))
                new_label2[1] = str(min(float(new_label2[1])+0.003125, 1))
                labels.append(new_label1)
                labels.append(new_label2)
        with open(os.path.join(new_label_path, file), 'w') as f:
            for i, label in enumerate(labels):
                labels[i] = ' '.join(label) + '\n'
            f.writelines(label1)


if __name__ == '__main__':
    label_path = '/home/zccao20215227103/hcxue/MedCV/test/label'
    new_label_path = '/home/zccao20215227103/hcxue/MedCV/test/new_label'
    data_balance(label_path, new_label_path)
