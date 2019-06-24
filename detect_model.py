import cv2
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def detect_one_face(img):
    result = detector.detect_faces(img)
    x1,y1,w,h = result[0]['box']
    face = img[y1:y1+h,x1:x1+w]
    face = cv2.resize(face,(160,160))
    return face

def detect_faces(frame):
    data = []
    result = detector.detect_faces(frame)
    thresh_hold = 0.9
    for item in result:
        if item['confidence'] > thresh_hold:
            x1,y1,w,h = item['box']
            x1 = abs(x1)
            y1 = abs(y1)
            w = abs(w)
            h = abs(h)
            face = frame[y1:y1+h,x1:x1+w]
            face = cv2.resize(face,(160,160))
            dic = {'face':face,'coor':(x1,y1,w,h)}
            data.append(dic)
    return data

if __name__ == '__main__':
    img = cv2.imread('/home/quanghuy/Source/FaceRecognition/tmp.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    data = detect_faces(img)
    print(len(data))