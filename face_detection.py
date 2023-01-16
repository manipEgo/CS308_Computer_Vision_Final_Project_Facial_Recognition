from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import re

db_path = 'F:/ThunderDownload/VGG-Face2/data/train'
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
model_name = models[0]
metrics = ["cosine", "euclidean", "euclidean_l2"]
metric_name = metrics[0]


def detect(img_path):
    recognition = DeepFace.find(img_path=img_path, db_path=db_path, model_name=model_name, distance_metric=metric_name, enforce_detection=False)
    print(recognition)
    result = recognition.to_numpy()
    
    
    img0 = plt.imread(img_path)
    plt.subplot(2, 2, 1)
    plt.imshow(img0)
    plt.title('origin img')
    plt.axis('off')

    for i in range(3):
        if i >= len(result): break
        imgi = plt.imread(result[i][0])
        namei = re.split('/|\\\\', result[i][0])[-2]
        plt.subplot(2, 2, i+2)
        plt.imshow(imgi)
        if i == 0: plt.title('best match\n name:'+namei)
        else: plt.title('match '+str(i+1)+'\n name:'+namei)
        plt.axis('off')

    plt.tight_layout(h_pad=1)
    plt.show()

if __name__ == '__main__':
    '''
    db_path: 数据库路径，其目录为各个以人名命名的文件夹，文件夹中为该人面部图片
    model_name: 选用的模型
    img_path: 用于识别的图片
    '''
    img_path = 'F:/ThunderDownload/VGG-Face2/data/test/114514/'
    pic_name = os.listdir(img_path)
    detect(img_path+pic_name[0])