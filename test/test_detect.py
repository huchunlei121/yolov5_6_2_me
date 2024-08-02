import torch
if __name__ == '__main__':
    # 直接加载github上的训练好的模型
    # 从github上下载项目代码到本地路径
    # 从代码中调用hubconf.py文件进行模型的加载恢复
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # Images 图像路径
    img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    # 推理预测
    results = model(img)
    # 结果展示
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()
