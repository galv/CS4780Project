import abc
from enum import Enum # We use enums to represent our chosen features.


"""
The main class for training various machine learning algorithms.
For every algorithm you want to train on the cifar-10 data set, you should
create a subclass of this class to minimize code reuse.
"""
class Trainer(object):
    __metaclass__ = abc.ABCMeta
    train_path = "data/train"
    test_path = "data/test"
    cv_path = "data/cross_validate"
    clf = None

    Feature = Enum('Feature', 'edge blob raw') #haar

    def main(self, feature = "raw"):
        #Train multiple models with different parameters.
        train_dicts = [f for f in os.listdir(os.path.join("data","train")) 
                       if path.isfile(f)]
        
        
    """
    
    """
    @abc.abstractmethod
    def initialize_model(self,**kwargs):

    """
    Trains the model clf on the training data set.
    """
    def train(self, X_train,y_train):
        

    def test(self, X_test, y_test):

    @abc.abstractmethod
    def cross_validate(self, X_cv, y_cv):

    
