import cPickle as pickle
import numpy as np
import os
#from scipy.misc import imread
from sklearn.cross_validation import train_test_split
import tensorflow as tf

TRAIN_VAL_SPLIT = 0.9
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000*TRAIN_VAL_SPLIT
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 50000*(1-TRAIN_VAL_SPLIT)
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def load_CIFAR_batch(filename, num):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['coarse_labels']
    X = X.reshape(num, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def preprocess_images(images):
    return tf.image.per_image_whitening(images)

def load_CIFAR100(ROOT):
  """ load all of cifar """
  Xtr, Ytr = load_CIFAR_batch(os.path.join(ROOT, 'train'),50000)
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test'),10000)
  Xtr = preprocess_images(Xtr)
  Xte = preprocess_images(Xte)
  Xtr, Xval, Ytr, Yval = train_test_split(Xtr, Ytr, test_size = TRAIN_VAL_SPLIT)
  return Xtr, Ytr, Xval, Yval, Xte, Yte



def get_batch(data, label, batch_size = 128, shuffle = True):
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * .4)
    images, label = tf.train.shuffle_batch([data, label],batch_size = batch_size, num_threads = 16, capacity = min_queue_example + batch_size*3, min_after_dequeue = min_queue_examples)
    return images, label

if __name__ == "__main__":
    Xtr, Ytr, Xval, Yval, Xte, Yte = load_CIFAR100(os.path.join(os.getcwd(),'cifar-100-python'))
    x_batch, y_batch = get_batch(Xtr, Ytr)

    print(x_batch)
    print(x_batch.shape)
    
    
