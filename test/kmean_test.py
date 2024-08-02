
from utils.autoanchor import kmean_anchors

if __name__ == '__main__':
    import os
    # os.chdir('..')  # 把当前路径设置成上一个文件夹，设置一次后改文件夹的当前路径都是上一级路径
    print(os.getcwd())
    k = kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)
    # k = kmean_anchors()

    print(k)
