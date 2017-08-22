
#define model
import keras
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Permute, UpSampling2D, AtrousConv2D
from keras.layers import Flatten, Input, BatchNormalization, Activation, merge, Reshape, ZeroPadding2D
from keras.models import Model, Sequential
import numpy as np

def conv_bn_relu(input_layer, filters, kernel_size, activation="relu", padding='same'):
    out = Conv2D(filters, (kernel_size, kernel_size), padding=padding)(input_layer)
    if activation is not None:
        out = BatchNormalization(axis=1)(out)
        out = Activation(activation)(out)
    return out

def getModel(C,H,W, classes, h=None, mode="train"):
    inp = Input(shape=(C,H,W))
    out = conv_bn_relu(inp, 64, 3)
    out = conv_bn_relu(out, 64, 3)
    out = MaxPooling2D(pool_size=(2,2))(out)
    
    out = conv_bn_relu(out, 128, 3)
    out = conv_bn_relu(out, 128, 3)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = conv_bn_relu(out, 256, 3)
    out = conv_bn_relu(out, 256, 3)
    out = conv_bn_relu(out, 256, 3)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = conv_bn_relu(out, 512, 3)
    out = conv_bn_relu(out, 512, 3)
    out = conv_bn_relu(out, 512, 3)
    if mode!="dense":
        if h is None: _,c,h,w = [ d.value for d in out.get_shape()]
        out = conv_bn_relu(out, 512, h, padding='valid')
        out = conv_bn_relu(out, classes, 1, activation=None, padding='valid')
    if mode=="train":
        out = Reshape((classes,-1))(out)
        out = Permute((2,1))(out)
        out = Activation("softmax")(out)
        out = Flatten()(out)
    if mode=="test":
        _,c,h,w = [ d.value for d in out.get_shape()]
        out = Reshape((classes,-1))(out)
        out = Permute((2,1))(out)
        out = Activation("softmax")(out)
        out = Permute((2,1))(out)
        out = Reshape((c,h,w))(out)
    
    if mode=="dense":
        out = MaxPooling2D(pool_size=(2,2))(out)
        out = Flatten()(out)
        out = Dense(512,activation='relu')(out)
        out = Dense(classes,activation='softmax')(out)
        #out = Permute((2,1))(out)
        #out = Activation("softmax")(out)
        ##out = Permute((2,1))(out)
        #out = Reshape((c,h,w))(out)
    #out = Flatten()(out)
    #out = Dense(512,activation="relu")(out)
    #out = Dense(classes, activation="softmax")(out)
    return Model(inp,out)
