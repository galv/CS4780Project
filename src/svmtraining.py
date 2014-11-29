#Train a decision tree
import os
import cPickle
import time
import math
from SimpleCV import *
from sklearn.svm import SVC
from sklearn import cross_validation

algo = "svm" 

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

def main(min_C = .0001, max_C = 1000, incrementor = 2, type = "raw", save = False):
    X_train, y_train = load("train", type)
    X_test, y_test = load("test", type)

    models = []
    current_C = min_C
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    total_runs = int(math.log(max_C/min_C * len(kernels), incrementor))
    current_run = 1
    avg_elapsed = 0.0
    while (current_C <= max_C):
        for kernel in kernels:
            start_time = time.time()
            print "Starting run " + str(current_run) + " out of run " + str(total_runs) + " at time " + str(start_time)
            classifier = SVC(C = current_C, kernel = kernel)
            average_accuracy = np.mean(cross_validation.cross_val_score(classifier, X_train, y_train, cv = 5))
            models.append((current_C, kernel, classifier, average_accuracy))
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_elapsed = ((current_run - 1) * avg_elapsed + elapsed_time) / current_run
            print "Ending run " + str(current_run) + " out of run " + str(total_runs) + " at time " + str(end_time)
            print "Elapsed time for this run was " + str(elapsed_time)
            print "Average elapsed time for all runs is " + str(avg_elapsed)
            current_run = current_run + 1
        current_C = current_C * incrementor

    models.sort(key = lambda x: x[3]) #WARNING: x[2] should correspond to average_accuracy of each model
    #Retrain one last time with optimal parameters so we can use all data.
    tuned_C = models[-1][0]
    tuned_kernel = models[-1][1]
    tuned_classifier = SVC(kernel = tuned_kernel, C = tuned_C)
    tuned_classifier.fit(X_train, y_train)
    accuracy = test(tuned_classifier, X_test, y_test)
    if save:
        import datetime
        cPickle.dump({"hypothesis" : tuned_classifier, "accuracy" : accuracy, "parameters" : {"C" : tuned_C, "kernel" : tuned_kernel}, "time" : datetime.datetime.now().time()}, open(os.path.join("results/", algo, type, type + ".p"), "wb"))
    return accuracy, tuned_classifier

def test(classifier, X, y_true):
    y_pred = classifier.predict(X)
    return np.sum(y_true == y_pred) #Returns accuracy
