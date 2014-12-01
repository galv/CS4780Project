import cPickle
import os
from SimpleCV import *
from sklearn.cluster import KMeans

def create_edges(imgs):
    edgeFeats = EdgeHistogramFeatureExtractor()
    # We must call nan_to_num because some images simply have no edges, 
    # which cause divison by zero, resulting in nans, which cannot be trained on.
    return map(lambda image: np.nan_to_num(edgeFeats.extract(image)), imgs)

def create_hues(imgs):
    hueFeats = HueHistogramFeatureExtractor()
    return map(hueFeats.extract, imgs)

def create_haar(imgs):
    haar = HaarLikeFeatureExtractor('util/haar.txt')
    return map (haar.extract, imgs)

def create_bag(imgs):
    kmeans = KMeans(verbose = 1, n_clusters = 10, n_jobs = 15)
    vectors = map (lambda x: x.getNumpy().flatten(), imgs)
    return kmeans.fit_transform(vectors)

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
        if False:
            X_feat = create_edges(X)
            cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "edge", "edge.p"), "wb"))
            print "edges done"
        if False:
            X_feat = create_hues(X)
            cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "hue", "hue.p"), "wb"))
            print "hues done"
        if False:
            X_feat = create_haar(X)
            cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "haar", "haar.p"), "wb"))
            print "haar done"
        if True:
            X_feat = create_bag(X)
            cPickle.dump((X_feat, y), open(os.path.join(data_path, type, "bag", "bag.p"), "wb"))
            print "kmeans done"
