import numpy as np
import matplotlib.pyplot as plt
import cv2
from mtcnn.mtcnn import MTCNN
import os

detector = MTCNN()

def load_image_and_crop_face(image_path,target_size=(160,160)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img)
    x1,y1,w,h = result[0]['box']
    x1 = abs(x1)
    y1 = abs(y1)
    face = img[y1:y1+h,x1:x1+w]
    face = cv2.resize(face,target_size)
    return face

def convert_to_tensor(X):
    temp = np.array(X[0])
    for i in range(1,len(X)):
        tmp = np.array(X[i])
        temp = np.concatenate((temp,tmp),axis=0)
    return temp

def create_dataset(image_path):
    if not os.path.exists(image_path):
            raise Exception('file not found')
    list_dir = os.listdir(image_path)
    list_absolute_dir = [os.path.join(image_path,x) for x in list_dir]
    list_image = []
    list_label = []
    for item in list_absolute_dir:
        people_image_name = os.listdir(item)
        list_absolute_path_image = [os.path.join(item,x) for x in people_image_name]
        list_face_one_people = [load_image_and_crop_face(x) for x in list_absolute_path_image]
        label_one_people = [item.split('/')[-1] for x in people_image_name]
        list_image.append(list_face_one_people)
        list_label.append(label_one_people)
    print(list_label)
    return convert_to_tensor(list_image),convert_to_tensor(list_label)

def save_tensor(X_train,y_train,X_val,y_val):
    if not os.path.exists('dataset'):
        try:
            np.savez_compressed('dataset',X_train,y_train,X_val,y_val)
        except Exception as e:
            print(e)
    return 

if __name__ == '__main__':
    train_path = '/home/quanghuy/Source/FaceRecognition/data/train'
    val_path = '/home/quanghuy/Source/FaceRecognition/data/val'
    X_train,y_train = create_dataset(train_path)
    X_val,y_val = create_dataset(val_path)
    #save_tensor(X_train,y_train,X_val,y_val)
    