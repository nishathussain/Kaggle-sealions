from scipy.misc import imsave, imresize
import argparse
from keras import optimizers
from keras.models import load_model

from polarbear import *

parser = argparse.ArgumentParser(description='Train on patch dataset')
parser.add_argument('--model', default="models/new_model.h5", help='model to load')
parser.add_argument('--H', default=1000, type=int, help='patch size')
parser.add_argument('--iid', default=0, type=int, help='iid to test')
args = parser.parse_args()


model = load_model(args.model)

ds = DataSource()
#val = ds.train[0]
aimg,_ = ds.dataset("val_scaled").aimg(args.iid)

from IPython.display import display
from PIL import Image
H = args.H
a1,a2,a3,a4 = aimg.cropHW()
#aimg = a4
for a,x,y in aimg.tile(H,H//2):
    W,H = a.WH
    X = np.expand_dims(a.np(), axis=0)
    Y = model.predict(X) 
    yy = (Y[0][0]*255).astype('uint8')
    img = Image.fromarray(yy) #.resize((W,H))
    #display(a)
    #display(img)
    img.save('tmp/'+str(x)+"_"+str(y)+"out.jpg")
    a.img.save('tmp/'+str(x)+"_"+str(y)+"in.jpg")
    #break
