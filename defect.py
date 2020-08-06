from dsvdd import *
import matplotlib.pyplot as plt
import scipy.io as io
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from dsvdd.utils import *
import numpy as np
import random
import warnings
from keras.preprocessing import image
from keras.preprocessing.image import *
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)



def main():
    random.seed(1)
    tf.reset_default_graph()
    from dsvdd.utils import plot_most_normal_and_abnormal_images
    #model = feature_extract1()  # 预训练网络特征提取
    # get dataset
    dataset = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    size = 512;  cut_size = 64;  p = 1
    for i in range(1):
        data_set = dataset[i]
        print('dataset:', data_set)
        print('INFO: Loading data...')
        path = 'model/' + data_set# + '_test1'
        if not os.path.exists(path):
            os.makedirs(path)
        x_train = get_traindata_cut(data_set, size, cut_size)
        #x_test, y_test = get_testdata_cut(data_set, size, cut_size)



        #svdd = DeepSVDD(input_shape=(cut_size, cut_size, 3))#
        #svdd.fit(x_train, x_train_label, x_test, x_test_label, y_test, path)


if __name__ == '__main__':
    main()
