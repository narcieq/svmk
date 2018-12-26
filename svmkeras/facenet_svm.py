import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skiamage.transform import resize
from keras.models import load_model

%matplotlib inline

cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'

image_dir_basepath = '../../../../VGG-Net/dataset/'
classes = ['high', 'middle', 'low']
image_size = 160

model_path = '../model/keras/model/facenet_keras.h5'
model = load_model(model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1,2,3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0,1,2)
        size = x.size
    else:
        raise ValueError('Dimension sould be 3 or 4')

    mean = np.mean(x,axis=axis, keepdims = True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x-mean) / std_adj
    return y

def l2_normalize(x,axis=-1, epsilon =1e-10):
    output = x/np.sqrt(np.maximum(np.sum(np.square(x), axis=axis,keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors=3)
        (x,y,w,h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                        x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images, batch_size)):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2l2_normalize(np.concatenate(pd))

    return embs

def train(dir_basepath):
    labels = []
    embs = []
    for class in classes:
        dirpath = os.path.abspath(dir_basepath + class)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)
        labels.extend([class]*len(embs_))
        embs.append(embs_)

    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernal='linear', probability=True).fit(embs, y)
    return le, clf

def infer(le, clf, filepaths):
    embs = calc_embs(filepaths)
    pred = le.inverse_transform(clf.predict(embs))

    return pred

le, clf = train(image_dir_basepath, classes)

test_dirpath = os.path.join(image_dir_basepath, 'Test')
test_filepaths = [os.path.join(test_dirpath, f) for f in os.listdir(test_dirpath)]

pred = infer(le, clf, test_filepaths)
