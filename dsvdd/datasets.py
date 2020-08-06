import numpy as np
import os
import cv2
import time
import random
import shutil
import scipy.io as io
from PIL import Image
from glob import glob
__all__ = ['get_traindata_cut', 'get_testdata_cut', 'get_testdata', 'get_traindata', 'defect_img', 'get_valdata']


@contextmanager
def task(_):
    yield

def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)
    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    return images


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))



def get_imgs(obj, mode):
    fpattern = os.path.join(DATASET_PATH, '{}/{}/*/*.png'.format(obj, mode))
    imgpaths = sorted(glob(fpattern))
    images = np.asarray(list(map(cv2.imread, imgpaths)))
    
    if images.shape[-1] != 3:
        images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images) / 255.
    return images


def generate_coords_position(size, cutsize):
    pos_to_diff = {
        0: (-1, -1),
        1: (-1, 0),
        2: (-1, 1),
        3: (0, -1),
        4: (0, 1),
        5: (1, -1),
        6: (1, 0),
        7: (1, 1)
    }

    with task('P1'):
        h1 = np.random.randint(0, size - cutsize + 1)
        w1 = np.random.randint(0, size - cutsize + 1)

    pos = np.random.randint(8)

    with task('P2'):

        K3_4 = 3 * cutsize // 4
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(cutsize // 4, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        h2 = np.clip(h2, 0, size - cutsize)
        w2 = np.clip(w2, 0, size - cutsize)

        p2 = (h2, w2)

    return p1, p2, pos




if __name__ == '__main__':
    np.random.seed = 1
    dataset = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    obj = dataset[0]
    DATASET_PATH = '../../MVTec_AD'
    starttime = time.time()


    x_train = get_imgs(obj, 'train')


    endtime = time.time()
    print('总共的时间为:', round(endtime - starttime, 2), 'secs')
    print(x_train.shape)


