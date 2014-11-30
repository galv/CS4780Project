import svmtraining as svm
import foresttraining as forest
import treetraining as tree
import knntraining as knn

def main():
    tree.main(type = "bag", save = True)
    tree.main(type = "haar", save = True)
    print "tree done"

    forest.main(type = "edge", save = True)
    forest.main(type = "hue", save = True)
    forest.main(type = "bag", save = True)
    forest.main(type = "haar", save = True)
    print "forest done"

    svm.main(type = "edge", save = True)
    svm.main(type = "hue", save = True)
    svm.main(type = "bag", save = True)
    svm.main(type = "haar", save = True)
    print "svm done"

    knn.main(type = "haar", save = True)
    knn.main(type = "bag", save = True)
    print "knn done"
