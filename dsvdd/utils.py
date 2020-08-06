from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.models import Model


__all__ = ['plot_most_normal_and_abnormal_images', 'task', 'ksdensity_ICDF', 'merge_image', 'feature_extract1', 'feature_extract']


def feature_extract(input_shape=(512, 512, 3), size=1):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    #pool1 = base_model.get_layer('block2_conv2').output
    #pool2 = base_model.get_layer('block3_conv4').output
    pool3 = base_model.get_layer('block4_conv4').output
    pool4 = base_model.get_layer('block5_conv4').output
    #pool1_feature = AveragePooling2D(pool_size=(size*8, size*8))(pool1)
    #pool2_feature = AveragePooling2D(pool_size=(size*4, size*4))(pool2)
    pool3_feature = AveragePooling2D(pool_size=(size*2, size*2))(pool3)
    pool4_feature = AveragePooling2D(pool_size=(size, size))(pool4)
    feature_out = concatenate([pool3_feature, pool4_feature], axis=-1)
    feature_out = GlobalAveragePooling2D()(feature_out)
    model = Model(inputs=base_model.input, outputs=feature_out)
    #model.summary()
    return model

def feature_extract1(input_shape=(64, 64, 3), size=1):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    pool1 = base_model.get_layer('block2_conv2').output
    pool2 = base_model.get_layer('block3_conv4').output
    pool3 = base_model.get_layer('block4_conv4').output
    pool4 = base_model.get_layer('block5_conv4').output
    pool1_feature = AveragePooling2D(pool_size=(size*8, size*8))(pool1)
    pool2_feature = AveragePooling2D(pool_size=(size*4, size*4))(pool2)
    pool3_feature = AveragePooling2D(pool_size=(size*2, size*2))(pool3)
    pool4_feature = AveragePooling2D(pool_size=(size, size))(pool4)
    feature_out = concatenate([pool1_feature, pool2_feature, pool3_feature, pool4_feature], axis=-1)
    feature_out = GlobalAveragePooling2D()(feature_out)
    model = Model(inputs=base_model.input, outputs=feature_out)
    #model.summary()
    return model

@contextmanager
def task(_=''):
    yield

def ksdensity_ICDF(x, p):
    '''Returns Inverse Kernel smoothing function at p points'''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF to get support values
    fint = interp1d(kde.cdf, kde.support)
    return fint(p)

def flatten_image_list(images, show_shape) -> np.ndarray:
    """

    :param images:
    :param tuple show_shape:
    :return:
    """
    N = np.prod(show_shape)

    if isinstance(images, list):
        images = np.array(images)

    for i in range(len(images.shape)):  # find axis.
        if N == np.prod(images.shape[:i]):
            img_shape = images.shape[i:]
            new_shape = (N,) + img_shape
            return np.reshape(images, new_shape)

    else:
        raise ValueError('Cannot distinguish images. imgs shape: %s, show_shape: %s' % (images.shape, show_shape))


def get_shape(image):
    shape_ = image.shape[-3:]
    if len(shape_) <= 1:
        raise ValueError('Unexpected shape: {}'.format(shape_))

    elif len(shape_) == 2:
        H, W = shape_
        return H, W, 1

    elif len(shape_) == 3:
        H, W, C = shape_
        if C in [1, 3]:
            return H, W, C
        else:
            raise ValueError('Unexpected shape: {}'.format(shape_))

    else:
        raise ValueError('Unexpected shape: {}'.format(shape_))


def merge_image(images, show_shape, order='row'):
    images = flatten_image_list(images, show_shape)
    H, W, C = get_shape(images)
    I, J = show_shape
    result = np.zeros((I * H, J * W, C), dtype=images.dtype)

    for k, img in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = img

    return result


def plot_most_normal_and_abnormal_images(X_test, score):
    fig, axes = plt.subplots(nrows=2)
    fig.set_size_inches((5, 5))
    inds = np.argsort(score)

    image1 = merge_image(X_test[inds[:10]], (2, 5))
    axes[0].imshow(np.squeeze(image1))
    axes[0].set_title('Most normal images')
    axes[0].set_axis_off()

    image2 = merge_image(X_test[inds[-10:]], (2, 5))
    axes[1].imshow(np.squeeze(image2))
    axes[1].set_title('Most abnormal images')
    axes[1].set_axis_off()
