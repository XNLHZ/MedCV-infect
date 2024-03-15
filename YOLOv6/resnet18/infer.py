import os

import shutil

import torch
import torchvision
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

from torchvision import transforms

from PIL import Image


from datetime import datetime

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def sp_class_infer(img):

    # 保存分类结果的路径：
    path_Nm = 'D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/output/cell_test_result/Nm'
    path_Eg = 'D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/output/cell_test_result/Eg'
    path_Vc = 'D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/output/cell_test_result/Vc'

    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # from PIL import Image, ImageFont, ImageDraw
    # # 导入中文字体，指定字号
    # font = ImageFont.truetype('SimHei.ttf', 32)

    idx_to_labels = np.load('D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/idx_to_labels.npy', allow_pickle=True).item()
    # print(idx_to_labels)

    model = torch.load('D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/checkpoint/best-0.960.pth')
    model = model.eval().to(device)



    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                        ])


    # image_path = 'dataset_original/test/total_data'
    out_path = 'output'

    check_dir(os.path.join(out_path, 'result'))
    check_dir(os.path.join(out_path, 'result', 'Nm'))
    check_dir(os.path.join(out_path, 'result', 'Vc'))
    check_dir(os.path.join(out_path, 'result', 'Eg'))
    # for image in os.listdir(image_path):
    #     img_pil = Image.open(os.path.join(image_path, image))

    # test
    # image_p = 'D:/pycharm/Medical-images/MedCV/YOLOv6/resnet18/output/2.jpg'
    # img_pil = Image.open(image_p)
    # img_pil = img_pil.convert('RGB')



    PIL_image = Image.fromarray(img)

    input_img = test_transform(PIL_image) # 预处理

    # input_img = test_transform(img_pil)  # 预处理

    input_img = input_img.unsqueeze(0).to(device)
    # 执行前向预测，得到所有类别的 logit 预测分数
    pred_logits = model(input_img)
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
    n = 1
    top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度
    # class_name = idx_to_labels[int(pred_ids)]  # 获取类别名称
    confidence = float(confs) * 100  # 获取置信度
    text = '{:>.4f}'.format(confidence)  # 保留 4 位小数

    if str(pred_ids) == '0' or confidence < 80:
        sp_class = 'Nm'
        PIL_image.save(os.path.join(path_Nm, 'Nm_' + text + ' ' + datetime.now().strftime("%H-%M-%S") + '.jpg'))
    elif str(pred_ids) == '1':
        sp_class = 'Vc'
        PIL_image.save(os.path.join(path_Vc, 'Vc_' + text + ' ' + datetime.now().strftime("%H-%M-%S") + '.jpg'))
    else:
        sp_class = 'Eg'
        PIL_image.save(os.path.join(path_Eg, 'Eg_' + text + ' ' + datetime.now().strftime("%H-%M-%S") + '.jpg'))
    return str(sp_class)




        # print(text)
        # if str(pred_ids) == '0':
        #     shutil.copyfile(os.path.join(image_path, image), os.path.join(out_path, 'result', 'Nm', image))
        # elif str(pred_ids) == '1':
        #     shutil.copyfile(os.path.join(image_path, image), os.path.join(out_path, 'result', 'Vc', image))
        # else:
        #     shutil.copyfile(os.path.join(image_path, image), os.path.join(out_path, 'result', 'Eg', image))


# sp_class_infer('')
