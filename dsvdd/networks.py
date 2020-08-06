
from tensorflow import keras
from keras.applications.vgg19 import VGG19  #引入vgg19
from keras.layers import *
from keras.models import Model
import h5py as h5py

__all__ = ['VGG']


def VGG(input_shape=(64, 64, 3)):
    base_model = VGG19(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False
    pool1 = base_model.get_layer('block2_conv2').output
    pool2 = base_model.get_layer('block3_conv4').output
    pool3 = base_model.get_layer('block4_conv4').output
    pool1_feature = AveragePooling2D(pool_size=(4, 4))(pool1)
    pool2_feature = AveragePooling2D(pool_size=(2, 2))(pool2)
    #pool3_feature = AveragePooling2D(pool_size=(8, 8))(pool3)
    feture = concatenate([pool2_feature, pool3], axis=-1, name='concatenate')
    #out_layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv0')(feture)
    output = GlobalAveragePooling2D(name='global_average_pooling2d')(feture)

    #output = Dense(128, activation='elu')(out)
    '''
    x = Dense(4*4*32, activation='elu')(output)
    x = Reshape(target_shape=(4, 4, 32), name='reshape')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 8
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 16
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 32
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    decoder_out = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    '''
    decoder = Dense(256, activation='relu')(output)
    decoder = Dense(256, activation='relu')(decoder)
    #decoder = Dense(512, activation='relu', name='dense3')(decoder)
    decoder_out = Dense(64*64*3, activation='sigmoid')(decoder)

    model = Model(inputs=base_model.input, outputs=output)
    decoder = Model(inputs=base_model.input, outputs=decoder_out)

    return model, decoder

