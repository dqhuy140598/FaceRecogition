from svm import *
from keras.models import load_model
import numpy as np
import cv2
from detect_model import *
import pickle
import argparse
import matplotlib.pyplot as plt
parse = argparse.ArgumentParser()
parse.add_argument('--video',help='Your Video Path')
parse.add_argument('--image',help='Your Image Path')
parse.add_argument('--output',help='Yout Output Video')

def get_embed_vec_train_val(model,X_train,X_val):
    train = model.predict(X_train)
    val = model.predict(X_val)
    return train,val

def normalize(X):
    return X.astype(np.float32)/255

def get_predict_on_image(image_path,embed_model,svm_model):
    if not os.path.exists(image_path):
        raise Exception('File Not Found')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    data = detect_faces(image)
    thresh_hold = 0.65
    font = cv2.FONT_HERSHEY_TRIPLEX
    for item in data:
        x1,y1,w,h = item['coor']
        face = image[y1:y1+h,x1:x1+w]
        face = cv2.resize(face,(160,160))
        y_true,y_pro = test_on_image(embed_model,svm_model,face)
        y_pro_final = format(y_pro,'.2f')
        if y_pro > thresh_hold:
            text = 'Person: '+y_true+', Proba: ' + str(y_pro_final)
        else:
            text = 'Person: Unknown'
        cv2.putText(image,text,(x1,y1-10), font, 1,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(image,(x1,y1),(x1+w,y1+h),(255,0,0),thickness=5)
    plt.imshow(image)
    plt.show()

def test_on_image(facenet,svm,img):
    clas = ['ben_afflek','elton_john','jerry_seinfeld','madonna','mindy_kaling']
    img = np.expand_dims(img,axis=0)
    img = normalize(img)
    embed_vec = facenet.predict(img)
    y_cls = svm.predict(embed_vec)
    y_pro = svm.predict_proba(embed_vec)
    y_true = clas[y_cls[0]]
    return y_true,y_pro.max()

def process_data(data,frame,svm_model,embed_model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thresh_hold = 0.65
    for item in data:
        x1,y1,w,h = item['coor']
        face = frame[y1:y1+h,x1:x1+w]
        face = cv2.resize(face,(160,160))
        y_true,y_pro = test_on_image(embed_model,svm_model,face)
        y_pro_final = format(y_pro,'.2f')
        if y_pro > thresh_hold:
            text = 'Person: '+y_true+', Proba: ' + str(y_pro_final)
        else:
            text = 'Person: Unknown'
        cv2.putText(frame,text,(x1,y1-10), font, 0.5,(125,0,125),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,0),thickness=5)
            
    return frame

def test_on_video(svm_model,embed_model,video_path):
    cap = cv2.VideoCapture(video_path)

    video_FourCC    = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps       = cap.get(cv2.CAP_PROP_FPS)
    video_size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True
    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*"MJPG"), video_fps,video_size)
    while(True):
        # Capture frame-by-frame

        ret, frame = cap.read()
        if frame is not None:
            data = detect_faces(frame)

            img = process_data(data,frame,svm_model,embed_model)
            # Our operations on the frame come here
            # Display the resulting frame
            cv2.imshow('frame',img)
            out.write(img)
        else:
            cv2.imshow('frame',frame)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(video_path,image_path=None):

    MODEL_SVM = os.path.join(os.getcwd(),'svm_model.sav')
    DATASET_PATH = os.path.join(os.getcwd(),'dataset.npz')
    MODEL_PATH = os.path.join(os.getcwd(),os.path.join('model','facenet_keras.h5'))
    WEIGHT_PATH = os.path.join(os.getcwd(),os.path.join('model','facenet_keras_weights.h5'))

    print('-- Start Loading Keras Facenet Model ')

    model = load_model(MODEL_PATH)
    model.load_weights(WEIGHT_PATH)

    print('--Load Keras Facenet Model Successfully --')

    if not os.path.exists(MODEL_SVM):
        
        X_train,y_train,X_val,y_val = load_dataset(DATASET_PATH)

        X_train = normalize(X_train)
        X_val = normalize(X_val)

        embed_vec_train,embed_vec_val = get_embed_vec_train_val(model,X_train,X_val)

        print('-- Start Trainning Support Vector Machine')
        
        svm_model = train_embedding_vector(embed_vec_train,y_train)

        print('-- Training Support Vector Machine Done ! --')
    
        print('-- Saving Support Vector Machine Model')

        pickle.dump(svm_model,open(MODEL_SVM,'wb'))

        print('-- Saving Done !')

    else:

        print('-- Loading Support Vector Machine Model')
        
        X_train,y_train,X_val,y_val = load_dataset(DATASET_PATH)

        X_train = normalize(X_train)
        X_val = normalize(X_val)

        svm_model = pickle.load(open(MODEL_SVM,'rb'))
    
        print('Loading Done !')

    if video_path is not None:
        if not os.path.exists(video_path):
            raise Exception('Video path not found !')
        test_on_video(svm_model,model,video_path)

    if image_path is not None:
        if not os.path.exists(image_path):
            raise Exception('File Not Found')
        get_predict_on_image(image_path,model,svm_model)


if __name__ =='__main__':

    args = parse.parse_args()
    video_path = args.video
    image_path = args.image
    main(video_path,image_path)


