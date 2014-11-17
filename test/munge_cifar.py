#A file to munge cifar data format to actual png images.

import SimpleCV.ImageClass as IC

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def munge(path):
    true = IC.Image('../data/train/1.png')
    attempt = unpickle('../data/cifar-10-batches-py/data_batch_1')['data'][0]
    attempt_img = IC.Image(attempt)
    return (true, attempt_img)
