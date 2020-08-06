import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import scipy.io as scio
from keras.preprocessing.image import *
import cv2
import time
from math import ceil, floor

def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)
    return fint(x)


def ksdensity_ICDF(x, p):
    '''Returns Inverse Kernel smoothing function at p points'''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF to get support values
    fint = interp1d(kde.cdf, kde.support)
    return fint(p)


if __name__ == "__main__":

    dataset = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    data = dataset[4]
    score = scio.loadmat('data/' + data + '/score.mat')
    score = score['score']
    label_img = scio.loadmat('data/' + data + '/label_img.mat')
    label_img = label_img['label_img']
    print(score.shape)
    size = np.int(np.sqrt(score.shape[-1]))
    print('size=', size)

    plt.figure()
    for i in range(7):
        plt.subplot(4, 7, i + 1)
        plt.imshow(array_to_img(label_img[i+20]), cmap='gray')  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.subplot(4, 7, i + 8)
        img = np.reshape(score[i+20], (size, size, 1))
        plt.imshow(array_to_img(img), cmap='gray')  # 显示图片
        plt.axis('off')  # 不显示坐标轴

    for i in range(7):
        plt.subplot(4, 7, i + 15)
        plt.imshow(array_to_img(label_img[i+70]), cmap='gray')  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.subplot(4, 7, i + 22)
        img = np.reshape(score[i+70], (size, size, 1))
        plt.imshow(array_to_img(img), cmap='gray')  # 显示图片
        plt.axis('off')  # 不显示坐标轴

    plt.show()


    '''
    dataFile = 'results/pred_val.mat'
    data = scio.loadmat(dataFile)
    data = np.squeeze(data['pred_val'])
    #print(data.shape)
    test = ksdensity_ICDF(data, 0.95)

    
    a     = np.array([1,2,3,4,5,7,8,2,10,6,8])
    label = np.array([1,0,0,0,0,0,1,1,1, 1,1])
    test =           [0,0,0,0,0,1,1,0,1, 1,1]
    R = 5
    test = (np.array(a > R))
    print(np.array(test))
    FDR = np.sum(test[np.nonzero(label)]) / test[np.nonzero(label)].shape  # 检出率
    FPR = np.sum(test[np.nonzero(-label + 1)]) / test[np.nonzero(-label + 1)].shape #误报率
    print('检出率:', FDR)
    print('误报率:', FPR)
    '''
    #for i in range(a.shape):



    #print(np.sum(b == np.array(a>R)))







