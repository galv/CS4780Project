#Train a decision tree
import os
import cPickle
import time
import numpy as np
from SimpleCV import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

algo = "knn" 

def run_all():
    features = ["edge", "hue"]#, "haar"]
    for feature in features:
        main(type = feature, save = True)

def load(data, feature):
    """
    Return (X, y), where X is the list of images (SimpleCV Image), and y is 
    the list of classifications.
    data: data of data: training or testing
    """
    #Settings
    train_path = os.path.join("data", data, feature) #put your image path here if you want to override current directory

    X = []
    y = []
    for f in os.listdir(train_path):
        (X_i, y_i) = cPickle.load(open(os.path.join(train_path,f), "rb"))
        if type(X_i) is np.ndarray(X_i):
            X_i = X_i.tolist()
        X = X + X_i #Append the two lists together
        y = y + y_i
    assert np.size(X,0) == 50000 or np.size(X,0) == 10000
    assert np.size(y) == 50000 or np.size(y) == 10000
    # Raws are stored as SimpleCV Images so they can easily be converted to
    # features using SimpleCV
    # Since machine learning aglorithms take feature vectors as inputs, we
    # flatten the underlying 3D matrices of the images here.
    if feature == "raw":
        X = map (lambda img: img.getNumpy().flatten(), X)
    return X,y

def main(max_neighbors = 20, type = "raw", save = False):
    X_train, y_train = load("train", type)
    X_test, y_test = load("test", type)

    models = []
    for neighbors in xrange(1,max_neighbors):
        classifier = KNeighborsClassifier(n_neighbors = neighbors)
        average_accuracy = np.mean(cross_validation.cross_val_score(classifier, X_train, y_train, cv = 5, n_jobs = 15, pre_dispatch = 15))
        models.append((neighbors, classifier, average_accuracy))
        print "done with: " + str(neighbors) + " out of " + str(max_neighbors)

    models.sort(key = lambda x: x[2]) #WARNING: x[2] should correspond to average_accuracy of each model
    #Retrain one last time with optimal parameters so we can use all data.
    tuned_neighbors = models[-1][0]
    tuned_classifier = KNeighborsClassifier(n_neighbors = tuned_neighbors)
    tuned_classifier.fit(X_train, y_train)
    accuracy = test(tuned_classifier, X_test, y_test)
    if save:
        import datetime
        cPickle.dump({"hypothesis" : tuned_classifier, "accuracy" : accuracy, "parameters" : {"neighbors" : tuned_neighbors}, "time" : datetime.datetime.now().time()}, open(os.path.join("results/", algo, type, type + ".p"), "wb"))
    return accuracy, tuned_classifier

def test(classifier, X, y_true):
    y_pred = classifier.predict(X)
    return np.sum(y_true == y_pred) #Returns accuracy
