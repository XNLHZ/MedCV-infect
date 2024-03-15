import os
import json

json_file_path = 'D:/pycharm/Medical-images/MedCV/dataset/original_trainset'
count_AB, count_KP, count_PA = 0, 0, 0
for file in os.listdir(json_file_path):
    if file.endswith('.json'):
        with open(os.path.join(json_file_path, file), "r", encoding="utf8") as fp:
            json_datas = json.load(fp)["shapes"]
        for data in json_datas:
            if data['label'] == '鲍曼':
                count_AB += 1
            elif data['label'] == '肺克':
                count_KP += 1
            else:
                count_PA += 1
print(count_AB, count_KP, count_PA)
