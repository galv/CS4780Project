#Train a decision tree
import os
import cPickle
import time
from SimpleCV import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation


"""
Return (X, y), where X is the list of images (SimpleCV Image), and y is the list of classifications.
type: type of data: training, testing, or cross validation.
"""
def load(type, feature):
    #Settings
    train_path = os.path.join("data", type, feature) #put your image path here if you want to override current directory

    X = []
    y = []
    for f in os.listdir(train_path):
        (X_i, y_i) = cPickle.load(open(os.path.join(train_path,f), "rb"))
        X = X + X_i #Append the two lists together
        y = y + y_i
    return X,y

def main(max_depth = 20):
    X_train, y_train = load("train", "edge")
    X_test, y_test = load("test", "edge")

    models = []
    for depth in xrange(1,max_depth):
        classifier = train(X_train, y_train, max_depth = depth)
        average_accuracy = np.mean(cross_validation.cross_val_score(classifier, X_train, y_train, cv = 4))
        models.append((depth, classifier, average_accuracy))
    
    models.sort(key = lambda x: x[2]) #WARNING: x[2] should correspond to average_accuracy of each model
    tuned_classifier = models[-1][1]
    accuracy = test(tuned_classifier, X_test, y_test)
    return accuracy, tuned_classifier

def train(X,y, max_depth):
    classifier = DecisionTreeClassifier(max_depth = max_depth)
    classifier.fit(X,y)
    return classifier

def test(classifier, X, y_true):
    y_pred = classifier.predict(X)
    return np.sum(y_true == y_pred) #Returns accuracy
