import os,time
from flask import Flask, request,render_template
from random import randint
from infer import infer

app = Flask(__name__)

@app.route("/upload",methods=["POST"])
def upload():
    #接受前端传送过来的文件
    file_obj = request.files.get("image")
    if file_obj is None:
        return render_template('./upload_infer.html', res="图片为空")
    # 保存
    image_file=os.path.join('./images', str(time.time_ns())+'.jpg')
    file_obj.save(image_file)
    # 执行推理
    result=infer(image_file)
    # 可删除，不删除则保留
    # os.remove(image_file)
    return render_template('./upload_infer.html', res=result)


@app.route("/",methods=["GET"])
def main():
    return render_template('./upload_infer.html', res="")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080,debug=True)
