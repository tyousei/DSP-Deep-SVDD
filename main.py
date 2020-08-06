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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)

def feature(data, model):
    features_train = []
    for i in range(data.shape[0]):
        pre_x = np.expand_dims(image.img_to_array(data[i]), axis=0)
        feature = np.squeeze(model.predict(pre_x))
        features_train.append(np.reshape(feature, (-1, feature.shape[-1])))
    return np.array(features_train)


def main():
    random.seed(1)
    tf.reset_default_graph()
    from dsvdd.utils import plot_most_normal_and_abnormal_images

    # get dataset
    dataset = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    size = 512
    p = 0.95
    filename = 'data/log.txt'
    with open(filename, 'w') as f:  # 如果filename不存在会自动创建,'w', 'a', 分别表示擦除原有数据再写入和将数据写到原数据之后
        f.write('log begin:\n')
    for ii in range(15):
        data_set = dataset[ii]
        print('-------------------------------------')
        print('-------------------------------------')
        print('dataset:', data_set)
        train_img = get_traindata(data_set, size)
        test_img, y_test, label_img = get_testdata(data_set, size)
        print('train_img shape:', train_img.shape)
        print('test_img shape:', test_img.shape)
        print('y_test shape:', y_test.shape)
        print('label_img shape:', label_img.shape)
        directory = 'data/' + data_set + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)
        io.savemat(directory + 'label_img.mat', {'label_img': label_img})
        # feature_extract
        model = feature_extract1(input_shape=(size, size, 3), size=1)
        train_data = np.concatenate(feature(train_img, model))
        x_test = feature(test_img, model)

        np.random.shuffle(train_data)
        num = train_data.shape[0]
        x_train = train_data[0:np.int(num*p)]
        x_val = train_data[np.int(num*p):-1]
        #x_train = train_data[0:-5000]
        #x_val = train_data[-5000:-1]

        #print('train_data shape:', train_data.shape)
        #print('train shape:', x_train.shape)
        #print('val shape:', x_val.shape)
        #print('test shape:', x_test.shape)

        # build model and DeepSVDD
        svdd = DeepSVDD(input_shape=x_train.shape[-1], representation_dim=128)
        # train DeepSVDD

        y_test = np.reshape(label_img, (-1, label_img.shape[-1]))
        y_test = np.concatenate(y_test)
        y_test = np.array(y_test > 0)
        auc = svdd.fit(x_train, x_val, np.concatenate(x_test), y_test, epochs=100, verbose=True)
        print('dataset-%2d, auc:  %.3f\n' % (ii, auc))
        with open(filename, 'a') as f:
            f.write('dataset-%2d, auc:  %.3f\n' % (ii, auc))
        # calculate R
        pred_val = np.array(svdd.predict(x_val))
        R = ksdensity_ICDF(pred_val, 0.99)
        pred_test = svdd.predict_test(x_test)
        score = []
        for i in range(pred_test.shape[0]):
            score.append(np.array(pred_test[i] > R))
        score = np.array(score)
        io.savemat(directory + 'pred_test.mat', {'pred_test': pred_test})
        io.savemat(directory + 'score.mat', {'score': score})



if __name__ == '__main__':
    main()
