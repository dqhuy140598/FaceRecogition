from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix 
import numpy as np
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

def load_dataset(dataset_path):
    dataset = np.load(dataset_path)
    return dataset['arr_0'],label_encoder(dataset['arr_1']),dataset['arr_2'],label_encoder(dataset['arr_3'])
    
def train_embedding_vector(X_train,y_train):
    svm = SVC(probability=True,kernel='linear')
    svm.fit(X_train,y_train)
    return svm

def evaluate_trainning(model,X_val,y_val):
    y_pre = model.predict(X_val)
    score = accuracy_score(y_val,y_pre)
    confusion = confusion_matrix(y_val,y_pre)
    return score,confusion

def predict_embedding_vec(model,embed_vec):
    y_cls = model.predict(embed_vec)
    y_prob = model.predict_proba(embed_vec)
    return y_cls,y_prob

def label_encoder(y):
    y_label = label.fit_transform(y)
    return y_label

if __name__ == '__main__':
    X_train,y_train,X_test,y_test = load_dataset('/home/quanghuy/Source/FaceRecognition/dataset.npz')
    