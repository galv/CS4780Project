#Train a decision tree
import os
import cPickle
import time
from SimpleCV import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
import datetime

algo = "tree" 

def run_all():
    features = ["edge", "hue", "haar"]
    for feature in features:
        main(type = feature, save = True)

def load(type, feature):
    """
    Return (X, y), where X is the list of images (SimpleCV Image), and y is 
    the list of classifications.
    type: type of data: training or testing
    """
    #Settings
    train_path = os.path.join("data", type, feature) #put your image path here if you want to override current directory

    X = []
    y = []
    for f in os.listdir(train_path):
        (X_i, y_i) = cPickle.load(open(os.path.join(train_path,f), "rb"))
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

def main(max_depth = 20, type = "raw", save = False):
    start_time = datetime.datetime.now()
    X_train, y_train = load("train", type)
    X_test, y_test = load("test", type)

    models = []
    for depth in xrange(1,max_depth):
        classifier = train(X_train, y_train, max_depth = depth)
        average_accuracy = np.mean(cross_validation.cross_val_score(classifier, X_train, y_train, cv = 5, n_jobs = -1))
        models.append((depth, classifier, average_accuracy))

    models.sort(key = lambda x: x[2]) #WARNING: x[2] should correspond to average_accuracy of each model
    #Retrain one last time with optimal parameters so we can use all data.
    tuned_depth = models[-1][0]
    tuned_classifier = train(X_train, y_train, tuned_depth)
    accuracy = test(tuned_classifier, X_test, y_test)
    if save:
        cPickle.dump({"hypothesis" : tuned_classifier, "accuracy" : accuracy, "parameters" : {"depth" : tuned_depth},"start_time" :start_time, "end_time" : datetime.datetime.now(), "total_time": datetime.datetime.now() - start_time}, open(os.path.join("results/", algo, type, type + ".p"), "wb"))
    return accuracy, tuned_classifier

def train(X,y, max_depth):
    classifier = DecisionTreeClassifier(max_depth = max_depth)
    classifier.fit(X,y)
    return classifier

def test(classifier, X, y_true):
    y_pred = classifier.predict(X)
    return np.sum(y_true == y_pred) #Returns accuracy
