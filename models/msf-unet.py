# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:14:27 2024

@author: Ming Gong
"""



import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras
from keras.models import *
from keras.layers import MaxPooling2D, BatchNormalization, Dropout, Conv2D, concatenate,UpSampling2D, Activation,Input,Add,Concatenate,ReLU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.utils import normalize, to_categorical 

from dataset.data_loader import dice_coef, bce_dice_loss, dice_loss

def conv(inputs,filters):
    conv1 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)
    
    conv2 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='he_normal')(act1)
    conv2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(conv2)

    
    return act2



def attention_block(fx,fg,inter_shape):
    
    shape_x = K.int_shape(fx)
    
    phi_x = Conv2D(inter_shape, (1, 1), padding='same')(fx)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(fg)
    concat_xg = Add()([phi_x,phi_g])
    act_xg = Activation("relu")(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(concat_xg)
    sigmoid_xg = Activation("sigmoid")(psi)
     
    y = multiply([sigmoid_xg, fx])
    
    

    #result_bn = BatchNormalization()(y)
    
    return y





def dilated_conv(inputs, filters, dilation_rate):
    # First convolution with dilation_rate of 1 (standard convolution)
    adjusted_input = Conv2D(filters, kernel_size=(1, 1), padding='same')(inputs)
    adjusted_input = BatchNormalization()(adjusted_input)
    
    conv1 = Conv2D(filters, 3, use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)
    
    # Second convolution with specified dilation_rate
    conv2 = Conv2D(filters, 3, use_bias=False, padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate)(act1)
    conv2 = BatchNormalization()(conv2)
    x = Add()([conv2, adjusted_input])
    
    act2 = Activation("relu")(x)


    return act2






def residual_unit(input_tensor,filters, kernel_size=3, strides=1, adjust_channels=False):
    """
    定义残差模块，可选地调整输入通道数
    """
    # 调整输入通道数和尺寸（如果需要）
    if adjust_channels:
        adjusted_input = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
        adjusted_input = BatchNormalization()(adjusted_input)
    else:
        adjusted_input = input_tensor

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # 将调整后的输入添加到最后的卷积层输出
    x = Add()([x, adjusted_input])
    x = Activation('relu')(x)
    return x


def PyramidPoolingModule(input_tensor, num_filters,filters):
    """
    Build a Pyramid Pooling Module for semantic segmentation.
    
    :param input_tensor: A 4D tensor, with shape (batch_size, height, width, channels)
    :param num_filters: Number of filters for the convolutional layers
    :return: Output tensor after the pyramid pooling module
    """

    # Base part for the pooling module
    pool1 = MaxPooling2D(pool_size=(1, 1))(input_tensor)
    pool2 = MaxPooling2D(pool_size=(2, 2))(input_tensor)
    pool3 = MaxPooling2D(pool_size=(4, 4))(input_tensor)
    pool4 = MaxPooling2D(pool_size=(8, 8))(input_tensor)
    
    # Convolution after pooling to reduce dimensions
    conv_pool1 = Conv2D(num_filters, 1, activation='relu')(pool1)
    conv_pool2 = Conv2D(num_filters, 1, activation='relu')(pool2)
    conv_pool3 = Conv2D(num_filters, 1, activation='relu')(pool3)
    conv_pool4 = Conv2D(num_filters, 1, activation='relu')(pool4)
    
    # Upsampling to original image size
    upsample1 = UpSampling2D(size=(1, 1), interpolation='bilinear')(conv_pool1)
    upsample2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv_pool2)
    upsample3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(conv_pool3)
    upsample4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(conv_pool4)
    
    # Concatenate the original feature map with the upsampled feature maps
    concatenated = Concatenate(axis=-1)([input_tensor, upsample1, upsample2, upsample3, upsample4])
    
    # Additional convolution can be added here if needed
    output = Conv2D(filters, 1, activation='relu')(concatenated)

    return output






def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    
    conv1 = conv(inputs,filters=32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #drop1 = Dropout(0.3)(pool1)
    
    conv2 = conv(pool1,filters=64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #drop2 = Dropout(0.3)(pool2)
    
    
    conv3 = residual_unit(pool2,filters=128,adjust_channels=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #drop3 = Dropout(0.3)(pool3)
    psp3 = PyramidPoolingModule(conv3,num_filters=64,filters=128)
    
    
    conv4 = residual_unit(pool3,filters=256,adjust_channels=True)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    psp4 = PyramidPoolingModule(conv4,num_filters=64,filters=256)
    
    dia_conv4 =  dilated_conv(pool3,filters=256,dilation_rate=2)
    dia_psp4 = PyramidPoolingModule(dia_conv4,num_filters=64,filters=256)
    cat4 = concatenate([psp4, dia_psp4],axis=3)
    
    
    
    
    conv5 = residual_unit(pool4,filters=512,adjust_channels=True)
    psp5 = PyramidPoolingModule(conv5,num_filters=128,filters=512)
    
    dia_conv5 =  dilated_conv(pool4,filters=512,dilation_rate=2)
    dia_psp5 = PyramidPoolingModule(dia_conv5,num_filters=128,filters=512)
    cat5 = concatenate([psp5,dia_psp5], axis=3)
    
    
    
    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(cat5))
    merge6 = concatenate([cat4,up6], axis = 3)
    conv6 = residual_unit(merge6,filters=256,adjust_channels=True)

    


    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([psp3,up7], axis = 3)
    conv7 = residual_unit(merge7,filters=128,adjust_channels=True)

    
    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = residual_unit(merge8,filters=64,adjust_channels=True)


    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = residual_unit(merge9,filters=32,adjust_channels=True)  

    

    
    conv9_1 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9_1)

    model = Model(inputs, conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer = opt, loss = bce_dice_loss, metrics = ['accuracy',dice_loss])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
