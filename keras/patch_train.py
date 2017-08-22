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
args = parser.parse_args()

D = args.D


from polarbear import *
ds = DataSource()
tgen = ds.get_patch_gen("train", d=D, batch_size=args.batch_size)
vgen = ds.get_patch_gen("val", d=D, batch_size=args.batch_size)

if args.load_model is None:
	model = getModel(3,D,D,2)
else: model = load_model(args.load_model)

model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=["accuracy"])

checkpoint = ModelCheckpoint('model.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='max')

#help(model.fit_generator)
model.fit_generator(tgen, steps_per_epoch=tgen.samples//args.batch_size, 
                    validation_data=vgen, validation_steps=vgen.samples//args.batch_size,
                    epochs=args.epochs, callbacks=[checkpoint])

