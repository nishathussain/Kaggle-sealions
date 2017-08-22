import os
import sys
import numpy as np
import torch
import keras
from keras.utils.np_utils import to_categorical

from polarbear import *

ds = DataSource()

img0 = ds.train.aimg(0)[0]

#Xt,Yt,Xv,Yv = img0.posnegHW(45)
Xt,Yt,_,_  = img0.posneg(45)

import numpy as np
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


#Xt,Yt = shuffle(Xt,Yt)
#Xv,Yv = shuffle(Xv,Yv)

_,C,H,W = Xt.shape

from model import *

model = getModel(C,H,W,2)

model.summary()

Xt = Xt/255.0
#Xv = Xv/255.0
Yt = to_categorical(Yt)
#Yv = to_categorical(Yv)

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])
#model.fit(Xt,Yt,nb_epoch)
#help(model.fit)
model.fit(Xt,Yt, nb_epoch=10, validation_split=0.1, batch_size=16, shuffle=True)
