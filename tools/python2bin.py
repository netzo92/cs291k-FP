#   Author: Metehan Ozten
#   Function: Convert file in python format to binary format
#   How to use: call: 'python python2bin.py DATASET_NAME'
import numpy as np
import cPickle as pickle
import os
import sys
import random

def open_py_data(file_name):
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        print(X.shape)
        print(Y.shape)
        X = np.array(X)
        Y = np.array(Y)
        Y = Y.reshape((Y.shape[0],1))
        labels_and_data = np.hstack((X,Y))
        return labels_and_data.astype('uint8')


def split_train_val_test(file_name, num_records):
    file_name_array = file_name.split('.')
    train_file_name = file_name_array[0]+'-train.bin'
    val_file_name = file_name_array[0]+'-val.bin'
    test_file_name = file_name_array[0]+'-test.bin'
    num_test = int(num_records*0.05) 
    num_val = int(num_records*0.05)
    num_train = num_records - num_val - num_test
    k = [0]*num_train+[1]*num_val+[2]*num_test
    random.shuffle(k)
    with open(file_name, 'rb') as ifile:
        with open(train_file_name, 'wb') as tr_f, open(val_file_name, 'wb') as val_f, open(test_file_name, 'wb') as test_f:
            for val in k:
                data = ifile.read(1+32*32*4)
                if val is 0:
                    tr_f.write(data)
                elif val is 1:
                    val_f.write(data)
                elif val is 2:
                    test_f.write(data)


def convert2bin(file_name):
    data = open_py_data(file_name)
    print(data.shape)
    print(file_name.split('.'))
    output_file_name = (file_name.split('.')[0]).strip()+'.bin'
    with open(output_file_name, 'wb') as f_out:
        data.tofile(f_out)
    split_train_val_test(output_file_name, data.shape[0])



        

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit("Not enough args")
    convert2bin(sys.argv[1])

