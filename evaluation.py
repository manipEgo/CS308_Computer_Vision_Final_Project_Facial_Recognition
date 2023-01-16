from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import re
import time

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
# model_name = models[0]
metrics = ["cosine", "euclidean", "euclidean_l2"]
# metric_name = metrics[0]
    

def detect(img_path, db_path, model, metric):
    start = time.time()
    recognition = DeepFace.find(img_path=img_path, db_path=db_path, model_name=model, distance_metric=metric, enforce_detection=False)
    end = time.time()
    result = recognition.to_numpy()
    return result, end - start

def eval(result, name):
    if len(result) == 0: return False
    name1 = re.split('/|\\\\', result[0][0])[-2]
    if len(result) == 1 and name == name1: return True
    if len(result) == 2 and name == name1: return True
    if len(result) >= 3:
        if name == name1: return True
        name2 = re.split('/|\\\\', result[1][0])[-2]
        name3 = re.split('/|\\\\', result[2][0])[-2]
        if name2 == name3 == name: return True
    return False


if __name__ == '__main__':
    db_path = 'F:/CVDataset/train'
    test_path = 'F:/CVDataset/test'

    grid_acc = [[0, 0, 0] for _ in range(8)]
    grid_time_train = [[0, 0, 0] for _ in range(8)]
    grid_time_test = [[0, 0, 0] for _ in range(8)]

    for i in range(len(models)):
        model = models[i]
        for j in range(len(metrics)):
            metric = metrics[j]
            
            corrects = 0
            total = 100
            total_time = 0
            training_time = 0
            b = True
        
            for dir in os.listdir(test_path):
                img = os.listdir(test_path + '/' + dir)[0]
                res, timecost = detect(test_path + '/' + dir + '/' + img, db_path, model, metric)
                if eval(res, dir):
                    corrects += 1
                if b:
                    training_time = timecost
                    b = False
                else: total_time += timecost

            grid_acc[i][j] = corrects / total
            grid_time_train[i][j] = training_time
            grid_time_test[i][j] = total_time

    print('accuracy')
    print(grid_acc)
    print('train time')
    print(grid_time_train)
    print('test time')
    print(grid_time_test)

    txt = open('result.txt','w')
    txt.write('accuracy')
    txt.write(str(grid_acc))
    txt.write('train time')
    txt.write(str(grid_time_train))
    txt.write('test time')
    txt.write(str(grid_time_test))
    txt.close()