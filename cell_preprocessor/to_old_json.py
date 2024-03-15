import json
import os


def type_check(file_path):
    type = []
    for file in os.listdir(file_path):
        os.rename(file_path + "/" + file, file_path + "/" + file.replace(" ", ""))
        file = file.replace(" ", "")
        if file.endswith(".json"):
            with open(file_path + "/" + file, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_json_file = {"flags": data["flags"], "imageData": data["imageData"], "imageHeight": data["imageHeight"],
                             "imageWidth": data["imageWidth"], "imagePath": data["imagePath"], "shapes": [],
                             "version": data["version"]}
            for cellData in data["shapes"]:
                if cellData["class"] not in type:
                    type.append(cellData["class"])
    print(type)

def format_trans(file_path, out_path):
    count_zxl, count_js, count_lb = 0, 0, 0
    for file in os.listdir(file_path):
        os.rename(file_path + "/" + file, file_path + "/" + file.replace(" ", ""))
        file = file.replace(" ", "")
        if file.endswith(".json"):
            with open(file_path + "/" + file, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_json_file = {"flags": data["flags"], "imageData": data["imageData"], "imageHeight": data["imageHeight"],
                             "imageWidth": data["imageWidth"], "imagePath": data["imagePath"], "shapes": [],
                             "version": data["version"]}

            type = ['中性粒细胞（正常）', '单核巨噬细胞（正常）', '淋巴细胞', '单核巨噬细胞（空泡）', '中性粒细胞（空泡）']
            for cellData in data["shapes"]:
                if cellData["class"] in type:
                    for index, point in enumerate(cellData["contours"]):
                        cellData["contours"][index] = cellData["contours"][index][0]
                    if cellData["class"] in ['中性粒细胞（正常）', '中性粒细胞（空泡）']:
                        obscure_label = '中性粒细胞'
                        color = '0'
                        # count_zxl += 1
                    elif cellData["class"] in ['单核巨噬细胞（正常）', '单核巨噬细胞（空泡）']:
                        obscure_label = '单核巨噬细胞'
                        color = '1'
                        # count_js += 1
                    else:
                        obscure_label = '淋巴细胞'
                        color = '2'
                        # count_lb += 1

                    up, down, left, right = 10000, 0, 10000, 0
                    for point in cellData["contours"]:
                        width, height = point[0], point[1]
                        if height < up:
                            up = height
                        elif height > down:
                            down = height
                        if width > right:
                            right = width
                        elif width < left:
                            left = width
                    center_point = [float('{:.6f}'.format((right - left) / 2.0 + left)), float('{:.6f}'.format((down - up) / 2.0 + up))]
                    points = [center_point]
                    if obscure_label == '中性粒细胞':
                        pass
                    elif obscure_label == '单核巨噬细胞':
                        pass
                        # if center_point[0] - (right - left)/6.0 > 0 and center_point[1] - (down - up)/6.0 > 0:
                        #     points.append([center_point[0] - (right - left)/6.0, center_point[1] - (down - up)/6.0])
                        # if center_point[0] - (right - left)/6.0 > 0 and center_point[1] + (down - up)/6.0 < data['imageHeight']:
                        #     points.append([center_point[0] - (right - left)/6.0, center_point[1] + (down - up)/6.0])
                        # if center_point[0] + (right - left)/6.0 < data['imageWidth'] and center_point[1] + (down - up)/6.0 < data['imageHeight']:
                        #     points.append([center_point[0] + (right - left)/6.0, center_point[1] + (down - up)/6.0])
                        # if center_point[0] + (right - left)/6.0 < data['imageWidth'] and center_point[1] - (down - up)/6.0 > 0:
                        #     points.append([center_point[0] + (right - left)/6.0, center_point[1] - (down - up)/6.0])
                    elif obscure_label == '淋巴细胞':
                        pass
                        # if center_point[1] - (down - up)/4.0 > 0:
                        #     points.append([center_point[0], center_point[1] - (down - up)/4.0])
                        # if center_point[1] + (down - up)/4.0 < data['imageHeight']:
                        #     points.append([center_point[0], center_point[1] + (down - up)/4.0])
                        # if center_point[0] - (right - left)/4.0 > 0:
                        #     points.append([center_point[0] - (right - left)/4.0, center_point[1]])
                        # if center_point[0] + (right - left)/4.0 < data['imageWidth']:
                        #     points.append([center_point[0] + (right - left)/4.0, center_point[1]])

                    for point in points:
                        newData = {"label": obscure_label, 'accurate_label': cellData["class"], 'color': color,
                                   "points": [point], "group_id": cellData["group_id"],
                                   "description": "", "shape_type": "points", "flags": {},
                                   'cell_width': '{:.6f}'.format(right - left), 'cell_height': '{:.6f}'.format(down - up)}
                        new_json_file["shapes"].append(newData)
                        if obscure_label == '中性粒细胞':
                            count_zxl += 1
                        elif obscure_label == '单核巨噬细胞':
                            count_js += 1
                        elif obscure_label == '淋巴细胞':
                            count_lb += 1

            with open(out_path + "/" + file, "w") as write_f:
                json.dump(new_json_file, write_f, indent=4, ensure_ascii=False)
    print(count_zxl, count_js, count_lb)


def main():
    file_path = "C:/Users/XHC/Desktop/医学图像/细胞标注/细胞标注数据/细胞标注_第二次数据/mask/test"
    out_path = "C:/Users/XHC/Desktop/医学图像/细胞标注/细胞标注数据/细胞标注_第二次数据/mask/old"
    format_trans(file_path, out_path)
    # type_check(file_path)


main()
