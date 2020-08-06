import tensorflow as tf
import cv2
import numpy as np
from math import ceil
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from dsvdd import *
from dsvdd.utils import *
from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import os
from keras.applications.vgg19 import VGG19  #引入vgg19
from keras.layers import *
from keras.models import Model
import keras.backend as K
import scipy.io as io
from .utils import task


class DeepSVDD:
    def __init__(self, input_shape, representation_dim=1408, batch_size=64, lr=1e-3):
        self.represetation_dim = representation_dim
        self.batch_size = batch_size
        #self.c = tf.get_variable('c', [self.represetation_dim], dtype=tf.float32, trainable=False)
        '''
        with tf.variable_scope('base_model'):
            base_model = VGG19(include_top=False, input_shape=input_shape, weights='imagenet')
            pool1 = base_model.get_layer('block2_conv2').output
            pool2 = base_model.get_layer('block3_conv4').output
            pool3 = base_model.get_layer('block4_conv4').output
            pool4 = base_model.get_layer('block5_conv4').output

            pool1_feature = AveragePooling2D(pool_size=(8, 8))(pool1)
            pool2_feature = AveragePooling2D(pool_size=(4, 4))(pool2)
            pool3_feature = AveragePooling2D(pool_size=(2, 2))(pool3)
            feature = concatenate([pool1_feature, pool2_feature, pool3_feature, pool4], axis=-1, name='concatenate')

            output = GlobalAveragePooling2D(name='global_average_pooling2d')(feature)


        with tf.variable_scope('train'):
            #x = Dense(256, activation='elu')(feature)
            #x = Dense(256, activation='elu')(x)
            #output = Dense(128, activation='elu')(feature)
            decoder = Dense(256, activation='elu')(output)
            decoder = Dense(512, activation='relu')(decoder)
            decoder_out = Dense(64*64*3, activation='sigmoid')(decoder)
            feature = Model(inputs=base_model.input, outputs=feature)
            self.model = Model(inputs=base_model.input, outputs=output)
            decoder = Model(inputs=base_model.input, outputs=decoder_out)
        '''
        with tf.variable_scope('train'):
            inputs = Input(shape=input_shape, name='input_1')
            x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='elu')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)
            x = Dense(1024, activation='elu')(x)
            x = BatchNormalization()(x)
            #x = Dense(128, activation='elu')(x)
            laten = BatchNormalization(name='out')(x)

            #x = Dense(256, activation='elu')(laten)
            #x = BatchNormalization()(x)
            decoder_out = Dense(1024, activation='sigmoid')(x)


            self.model = Model(inputs=inputs, outputs=laten)
            decoder = Model(inputs=inputs, outputs=decoder_out)

        with task('Build graph'):
            # build network
            self.x = tf.placeholder(tf.float32, [None]+list(input_shape))  #[None]+list(input_shape)
            self.y = tf.placeholder(tf.float32, [None, 1024.])
            '''
            self.laten = tf.layers.dense(self.x, units=128, activation=tf.nn.elu)
            x = tf.layers.dense(self.laten, units=256, activation=tf.nn.elu)
            self.outputs = tf.layers.dense(x, units=input_shape, activation=tf.nn.sigmoid)
            
            decoder.summary()
            trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'train')
            for i in trainable_var:
                print(i)
            '''

            self.laten, self.outputs = self.model(self.x), decoder(self.x)
            #self.reconstruction = tf.compat.v1.losses.mean_squared_error(self.outputs, self.x)
            self.reconstruction = tf.reduce_sum(tf.square(self.outputs - self.y), -1)
            #self.reconstruction = -tf.image.ssim(decode, self.x, 1)
            self.dist_op = tf.reduce_sum(tf.square(self.laten), axis=-1)
            self.score_op = self.reconstruction # + self.reconstruction * 0.2
            self.loss_op = self.reconstruction

            '''
            opt = tf.train.GradientDescentOptimizer(1e-4)
            self.train_op1 = opt.minimize(self.reconstruction)
            self.train_op = opt.minimize(self.loss_op)
            '''
            self.train_op1 = tf.train.AdamOptimizer(1e-3).minimize(self.reconstruction) #, var_list=trainable_var
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss_op)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def fit(self, X,  Y, X_test, Y_test, y_test, path, epochs=200, verbose=True):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        self.sess.run(tf.global_variables_initializer())

        #self.pretraining(X)
        #self._init_c(X, path)

        #opt = tf.train.GradientDescentOptimizer(1e-4)
        #self.train_op = opt.minimize(self.loss_op)
        ops = {
            'train': self.train_op,
            'res': tf.reduce_mean(self.reconstruction),
            'dist': tf.reduce_sum(self.dist_op)
        }
        keras.backend.set_learning_phase(True)
        a = 0
        model_path = path + '/SVDD.h5'
        print('INFO: Save modol to path: ', model_path)
        for i_epoch in range(epochs):
            ind = np.random.permutation(N)
            x_train = X[ind]
            y_train = Y[ind]
            g_batch = range(BN) if verbose else range(BN)
            dist, res = 0, 0
            for i_batch in g_batch:
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                y_batch = y_train[i_batch * BS: (i_batch + 1) * BS]
                result = self.sess.run(ops, feed_dict={self.x: x_batch, self.y: y_batch})
                dist += result['dist']
                res += result['res']
            #R = ksdensity_ICDF(pred_val, 0.99)
            pred_test = np.squeeze(self.predict(X_test, Y_test))
            #print(X_test.shape, pred_test.shape)
            auc = roc_auc_score(y_test, pred_test)
            print('Epoch:%3d/%3d, dist:, %.4f, res:, %.4f, auc:%.3f' % (i_epoch + 1, epochs, dist, res / N, auc))
            '''
            if(auc > a):
                #print('INFO: Save best model')
                print('Epoch:%3d/%3d, dist:, %.4f, res:, %.4f, auc:%.3f' % (i_epoch + 1, epochs, dist, res / N, auc))
                self.model.save(model_path)
                a = auc
            '''

    def predict(self, X, Y):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        scores = list()
        keras.backend.set_learning_phase(False)
        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            y_batch = Y[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.sess.run(self.score_op, feed_dict={self.x: x_batch, self.y: y_batch})
            scores.append(s_batch)
        return np.concatenate(scores)


    '''
    def _init_c(self, X, path, eps=1e-1):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        keras.backend.set_learning_phase(1)
        print('INFO: Init the center C...')
        with task('1. Get output'):
            latent_sum = np.zeros(self.laten.shape[-1])
            epochs = 5

            ops = {'train': self.train_op1, 'res': tf.reduce_mean(self.reconstruction)}
            for i_epoch in range(epochs):
                ind = np.random.permutation(N)
                x_train = X[ind]
                res = 0
                for i_batch in range(BN):
                    x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                    result = self.sess.run(ops, feed_dict={self.x: x_batch})
                    res = result['res'] + res
                print('Epoch:%3d/%3d, loss:%.5f' % (i_epoch + 1, epochs, res/N))

            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]
                latent_v = self.sess.run(self.laten, feed_dict={self.x: x_batch})
                latent_sum += latent_v.sum(axis=0)
            c = latent_sum / N

        with task('2. Modify eps'):
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps
        c_path = path + '/c.mat'
        io.savemat(c_path, {'c': c})
        print('INFO: Save C to path: ', c_path)
        print('INFO: The center C have been initialized ...')
        self.sess.run(tf.assign(self.c, c))
    '''