# -*- coding=utf-8 -*-
import keras
import cv2
import os
import warnings
import numpy as np
import scipy.io as io
import scipy.io as scio
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from dsvdd.datasets import *
from dsvdd.utils import *
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from keras.preprocessing import image
from keras.models import Model
#from skimage import exposure, data
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)


def segmentation(test, feature_img, model, resize):
    class_output = tf.reduce_sum(model.output ** 2, -1)
    gap_weights = model.get_layer("input_1")
    grads = K.gradients(class_output, gap_weights.output)[0]
    iterate = K.function([model.input], [grads])
    pooled_grads_value = iterate([test])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)

    feature_map, heatmap = [], []
    for ii in range(test.shape[0]):
        for jj in range(1024):
            feature_img[ii, :, :, jj] *= pooled_grads_value[ii, jj]

        heat = np.mean(feature_img[ii], axis=-1)
        heat = np.maximum(heat, 0)  # relu激活
        heat /= np.max(heat)
        if resize:
            heat = cv2.resize(heat, (64, 64))
        heat = np.uint8(255 * heat)
        feature_map.append(heat)
        heatmap.append(cv2.applyColorMap(heat, cv2.COLORMAP_JET))
    return np.array(feature_map), np.array(heatmap)

def seg(img, label_img):

    threshold = 148
    print('threshold = ', threshold)
    img = np.maximum(img, threshold)
    img[img == threshold] = 1
    img[img > threshold] = 0
    return np.array(img)


if __name__ == '__main__':

    K.set_learning_phase(1)  # set learning phase

    dataset = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    data_set = dataset[0]
    cut_size = 64
    path = 'model/' + data_set + '_test1'
    model_img = feature_extract(input_shape=(cut_size, cut_size, 3))
    model_img.summary()
    #val_img = get_valdata(data_set, 512)
    test_img, label_img = get_testdata(data_set, 512, cut_size)
    print(test_img.shape, label_img.shape)
    # 获取特征图（32*32*1408）
    feature_layer_model = Model(inputs=model_img.input,
                               outputs=model_img.get_layer('concatenate_1').output)
    #feature_val_img = feature_layer_model.predict(val_img)
    feature_test_img = feature_layer_model.predict(test_img)
    print(feature_test_img.shape)
    # 获取GAP层输出
    #feature_val = model_img.predict(val_img)
    feature_test = model_img.predict(test_img)

    model = keras.models.load_model(path + '/SVDD.h5')
    model.summary()


    feature_laten = Model(inputs=model.input,
                                outputs=model.get_layer('batchNormalization').output)
    feature_test_laten = feature_laten.predict(feature_test)
    print(feature_test_laten[0:20][0:10])


    #feature_val_map, heatmap_val = segmentation(feature_val, feature_val_img, model, 0)
    feature_test_map, heatmap_test = segmentation(feature_test, feature_test_img, model, 1)
    #print(feature_val_map.shape, feature_test_map.shape)

    pre_flat = feature_test_map.reshape(-1)
    label_flat = label_img.reshape(-1)
    auc = roc_auc_score(np.array(label_flat).astype(int), pre_flat)
    print('auc = %.4f' % auc)
    a = 0
    if a == 1:
        #threshold = ksdensity_ICDF(feature_val_map.reshape(-1), 0.05)
        threshold = 123
        print(threshold)
        pre_flat[pre_flat <= threshold] = 1
        pre_flat[pre_flat > threshold] = 0

        sum = pre_flat + label_flat
        gother = np.sum(np.array(sum == 2))
        union = np.sum(np.array(sum == 1)) + gother
        IOU = round(gother / (union + 0.001), 3)
        print('IOU = %.4f' % IOU)

    '''
    IOU = 0
    for i in range(test_img.shape[0]):
        sum_img = feature_test_map[i] + label_img[i]
        gother = np.sum(np.array(sum_img == 2))
        union = np.sum(np.array(sum_img == 1)) + gother
        iou = round(gother / (union + 0.001), 3)
        IOU += iou
        # print('id-%3d, %5d, %5d, %.3f' % (i, gother, union, iou))
    IOU = IOU / test_img.shape[0]
    '''

    #print(feature_map.shape, heatmap.shape)
    #io.savemat(path + '/heatmap.mat', {'feature_map': feature_map, 'heatmap': heatmap})


    n = 10
    plt.figure()
    for i in range(10):
        plt.subplot(4, 10, i+1)
        plt.imshow(test_img[i+n][:, :, ::-1])  #[:, :, ::-1]
        #plt.imshow(test_img[i + n][:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.subplot(4, 10, i+11)
        plt.imshow(heatmap_test[i+n], cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(4, 10, i + 21)
        plt.imshow(test_img[i+n+10][:, :, ::-1])  # [:, :, ::-1]
        #plt.imshow(test_img[i + n + 10][:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.subplot(4, 10, i + 31)
        plt.imshow(heatmap_test[i+n+10], cmap='gray')
        plt.xticks([]), plt.yticks([])

    plt.show()








    '''
    plot_roc(img, label_img)

    print(seg_img.shape, label_img.shape)
    
    
    
    print('IOU = %.3f'% IOU)
    plt.subplot(1, 2, 1)
    plt.imshow(label_img[0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img[0], cmap='gray')
    plt.show()





    
    img_orig = test_img[0][:, :, ::-1]
    label = label_img[0]
    heat = heatmap[0]
    for i in range(9):
        img_orig = np.hstack((img_orig, test_img[i+1][:, :, ::-1]))
        label = np.hstack((label, label_img[i + 1]))
        heat = np.hstack((heat, heatmap[i + 1]))

    fig, axes = plt.subplots(nrows=3)
    fig.set_size_inches((5, 5))
    axes[0].imshow(img_orig)
    axes[0].set_axis_off()
    axes[1].imshow(label)
    axes[1].set_axis_off()
    axes[2].imshow(heat, cmap='gray')
    axes[2].set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    '''