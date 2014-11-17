import os
import glob
import time
from SimpleCV import *
from sklearn.tree import DecisionTreeClassifier

def load():
    #Settings
    my_images_path = "data/train" #put your image path here if you want to override current directory
    extension = "*.png"

    #Program
    path = my_images_path
    imgs = list() #load up an image list
    directory = os.path.join(path, extension)
    files = glob.glob(directory)
    for file in files[0:10000]:
        imgs.append(Image(file))
    #Get classifications
    y = np.loadtxt('data/train/trainLabels.csv', delimiter = ',', dtype = np.str,skiprows = 1)
    y = y[:,1]
    return imgs, y

def get_features(imgs):
    edgeFeats = EdgeHistogramFeatureExtractor()
    # We must call nan_to_num because some images simply have no edges, 
    # which cause divison by zero, resulting in nans, which cannot be trained on.
    return map(lambda image: np.nan_to_num(edgeFeats.extract(image)), imgs)

def main():
    train_imgs, labels = load()
    train_feats = get_features(train_imgs[0:10000])
    clf = train(train_feats, labels[0:10000])
    return clf

def train(X,y):
    clf = DecisionTreeClassifier()
    clf.fit(X,y)
    return clf

def test(clf, X, y_true):
    y_pred = clf.predict(X)
    return np.sum(y_true = y_pred)
