import abc
"""
The main class for training various machine learning algorithms.
For every algorithm you want to train on the cifar-10 data set, you should
create a subclass of this class to minimize code reuse.
"""
class Trainer(object):
    __metaclass__ = abc.ABCMeta

    def main():
    #Train multiple models with different parameters.

    """
    
    """
    def initialize_model(**kwargs):

    """
    Trains the model clf on the training data set.
    """
    def train(clf, X_train,y_train):
        

    def test(clf, X_test, y_test):

    @abc.abstractmethod
    def cross_validate(clf, X_cv, y_cv):

    
