#A file to munge cifar data format to 3D matrices.
import SimpleCV.ImageClass as IC
import numpy as np
import os
import cPickle

def main():
    raw_data_path = os.path.join("data","cifar-10-batches-py")
    data_files = [ f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f)) and "_batch" in f ]
    data_files.sort()
    count = 1
    for data_file in data_files[0:1]:
        image_package = unpickle(os.path.join(raw_data_path,data_file))
        for binary_image in image_package["data"]:
            print count
            formatted_image = munge(binary_image)
            cv_image = IC.Image(formatted_image)
            cv_image.save(os.path.join("data", "cifar-10-png", str(count) + ".png"))
            count += 1

# Dict train or test, which contains X and y, where y is labels and X is images.
def save(file_name, dict):
    fi = open(file_name, 'rb')
    cPickle.dump(dict, fi)
    fi.close()

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# We need to turn the CIFAR-10 format into a 3d matrix of size 32*32*3 so 
# that we SimpleCV can open them with the "Image()" constructor.
def munge(cifar_image):
    #true = IC.Image('data/train/1.png')
    #true_array = true.getNumpy()[:,:,:]
    #true_red = true_array[:,:,0]
    #true_green = true_array[:,:,1]
    #true_blue = true_array[:,:,2]
    attempt = cifar_image
    original_red = attempt[0:1024]
    original_green = attempt[1024:2048]
    original_blue = attempt[2048:3072]
    attempt_array = np.ndarray((32,32,3), np.uint8)

    for i in xrange(32):#0-31, rows
        for j in xrange(32):#0-31, columns
            attempt_array[i,j,0] = original_red[32*i + j] #Red
            attempt_array[i,j,1] = original_green[32 * i + j] #Green
            attempt_array[i,j,2] = original_blue[32 * i + j] #Blue
    red = attempt_array[:,:,0].T
    green = attempt_array[:,:,1].T
    blue = attempt_array[:,:,2].T
    attempt_array[:,:,0] = red
    attempt_array[:,:,1] = green
    attempt_array[:,:,2] = blue
    #assert attempt_array.all() == true_array.all()
    attempt_image = IC.Image(attempt_array)
    return attempt_array
