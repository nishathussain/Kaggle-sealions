from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from model import *
from keras.models import load_model
import argparse
from keras import optimizers

parser = argparse.ArgumentParser(description='Train on patch dataset')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--name', default="model", help='name of the exp')
parser.add_argument('--D', default=90, type=int, help='patch size')
parser.add_argument('--load_model', default=None, help='pretrained model')
parser.add_argument('--mid', default=True, type=bool, help='use mid patches')
parser.add_argument('--iid', default=70, type=int, help='iid to train')
parser.add_argument('--mode', default="train", help='mode of model')
parser.add_argument('--split', default=False, type=bool, help='split train and val')
parser.add_argument('--sep', default=None, type=int, help='seperation for mid patches')
args = parser.parse_args()

D = args.D
print(args)
import numpy as np
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


from polarbear import *
ds = DataSource()
img0,_ = ds.dataset("val_scaled").aimg(args.iid)
img0 = img0.fpups()
#if args.iid == 70: img0 = img0.crop(1500,1500,3000,4000)
args.split=False
if args.split:
    print("splitting")
    if args.mid: 
        Xt1,Yt1,Xv1,Yv1 = img0.posmidHW(D//2)
        Xt2,Yt2,Xv2,Yv2 = img0.posnegHW(D//2)
        Xt,Xv = np.concatenate([Xt1,Xt2], axis=0), np.concatenate([Xv1,Xv2], axis=0)
        Yt,Yv = np.concatenate([Yt1,Yt2], axis=0), np.concatenate([Yv1,Yv2], axis=0)
    else: 
        Xt,Yt,Xv,Yv = img0.posnegHW(D//2)
else:
    if args.mid:
        Xt1,Yt1,_,_ = img0.posmid(D//2, args.sep)
        Xt2,Yt2,_,_ = img0.posneg(D//2)
        Xt = np.concatenate([Xt1,Xt2], axis=0)
        Yt = np.concatenate([Yt1,Yt2], axis=0)
        Xv,Yv = Xt,Yt
    else: 
        Xt,Yt,_,_ = img0.posneg(D//2)
        Xv,Yv = Xt,Yt

#tgen = ds.get_patch_gen("train", d=D, batch_size=args.batch_size)
#vgen = ds.get_patch_gen("val", d=D, batch_size=args.batch_size)
#Xt,Yt = shuffle(Xt,Yt)
#Xv,Yv = shuffle(Xv,Yv)

_,C,H,W = Xt.shape

if args.load_model is None:
	model = getModel(C,D,D,2,mode=args.mode)
else: model = load_model(args.load_model)
    
from keras.utils.np_utils import to_categorical
Yt = to_categorical(Yt)
Yv = to_categorical(Yv)
    
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=["accuracy"])

checkpoint = ModelCheckpoint('model_'+str(args.iid)+'.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='max')

#help(model.fit_generator)
#model.fit_generator(tgen, steps_per_epoch=tgen.samples//args.batch_size, 
#                    validation_data=vgen, validation_steps=vgen.samples//args.batch_size,
#                    epochs=args.epochs, callbacks=[checkpoint])

print("training" , Xt.shape)
print("testing" , Xv.shape)
model.fit(Xt, Yt, batch_size=args.batch_size, epochs=args.epochs, validation_data=(Xv,Yv),  callbacks=[checkpoint], shuffle=True)
