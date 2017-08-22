import argparse
from keras.models import load_model
from model import *

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('--inp', default="models/model_zero.h5", help='input model')
parser.add_argument('--out', default="models/new_model_zero.h5", help='out model')
parser.add_argument('--H', default=1000, type=int, help="input size of output model")
parser.add_argument('--hh', default=11, type=int, help="size of previous model")
parser.add_argument('--mode', default="test", help="type of model")
args = parser.parse_args()



def save_model(inp="model.h5", out="new_model.h5", H=None, mode="none"):
    model = load_model(inp)
    new_model = getModel(3, H, H, 2, h=args.hh, mode=mode)
    for i in range(len(new_model.layers)):
        if len(new_model.layers[i].weights)!=0:
            new_model.layers[i].set_weights(model.layers[i].get_weights())
    #new_model.summary()
    if out is not None: new_model.save(out)
    return new_model


if __name__ == "__main__":
    save_model(args.inp, args.out, args.H, mode=args.mode)
