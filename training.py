import os
import glob
import time
from SimpleCV import *
from sklearn.tree import DecisionTreeClassifier

def load():
    #Settings
    my_images_path = "train" #put your image path here if you want to override current directory
    extension = "*.png"

    #Program
    path = my_images_path
    imgs = list() #load up an image list
    directory = os.path.join(path, extension)
    files = glob.glob(directory)
    for file in files[0:10]:
        imgs.append(Image(file))
    #Get classifications
    y = np.loadtxt('trainLabels.csv', delimiter = ',', dtype = np.str,skiprows = 1)
    y = y[:,1]
    return imgs, y

def get_features(imgs):
    edgeFeats = EdgeHistogramFeatureExtractor()
    return map(edgeFeats.extract, imgs)

def main():
    train_imgs, labels = load()
    train_feats = get_features(train_imgs[0:10])
    clf = train(train_feats, labels[0:10])
    return clf

def train(X,y):
    clf = DecisionTreeClassifier()
    clf.fit(X,y)
    return clf

def test(clf, X, y_true):
    y_pred = clf.predict(X)
    return np.sum(y_true = y_pred)
