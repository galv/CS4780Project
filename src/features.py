import cPickle
import os
from SimpleCV import *

def create_edges(imgs):
    edgeFeats = EdgeHistogramFeatureExtractor()
    # We must call nan_to_num because some images simply have no edges, 
    # which cause divison by zero, resulting in nans, which cannot be trained on.
    return map(lambda image: np.nan_to_num(edgeFeats.extract(image)), imgs)

def create_hues(imgs):
    hueFeats = HueHistogramFeatureExtractor()
    return map(hueFeats.extract, imgs)

def create_bag_of_features(imgs):
    bagFeats = BOFFeatureExtractor(patchsz= (5,5), imglayout= (4,8), numcodes = 32)
    #Need to create codebook
    return map(bagFeats.extract, imgs)

def make_features(data_path = "data/"):
    types = ["train","test"]
    for type in types:
        X = []
        y = []
        data_files = os.listdir(os.path.join(data_path, type, "raw"))
        for f in data_files:
            (X_i, y_i) = cPickle.load(open(os.path.join(data_path, type, "raw", f), "rb"))
            X = X + X_i #Append the two lists together
            y = y + y_i
        X_feat = create_edges(X)
        cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "edge", "edge.p"), "wb"))
        print "edges done"
        X_feat = create_hues(X)
        cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "hue", "hue.p"), "wb"))
        print "hues done"
        #X_feat = create_bag_of_features(X)
        #cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "bag", "bag.p"), "wb"))
