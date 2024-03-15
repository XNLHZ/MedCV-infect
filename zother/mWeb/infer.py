import random

model_file=""
def infer(image_file):
    # load model  
    # load image from iamge_file
    # 可能要处理图片如正则化
    # exec infer: result = model(image)
    # return result
    return ["没病",'有病'][random.randint(0,1)]
