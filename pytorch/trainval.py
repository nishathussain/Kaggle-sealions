from __future__ import print_function
import cv2
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import SLv, v2, v1, detection_collate, BaseTransform
from sealions import SLroot, SLAnnotationTransform, SLDetection
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import time
from tqdm import trange, tqdm
import numpy as np
from sealions import SLroot, SL_CLASSES as labelmap
from polarbear import *
from PIL import Image
from scipy.misc import imsave, imread

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='SLv', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--th', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--exp_name', default='seal', help='Location to exp log data')
parser.add_argument('--num_classes', default=6, type=int, help='number of classes')
parser.add_argument('--trained_model', default=None, help='starting trained model')
parser.add_argument('--type', default='non_zero_ann', help='type of subset')
parser.add_argument('--fg', default=True, type=bool, help='single class')
parser.add_argument('--alpha', default=0.1, type=float, help='loss alpha')
parser.add_argument('--neg_th', default=0.1, type=float, help='neg threshold')


args = parser.parse_args()

cfg = (v1, v2)[args.version == 'v2']
if args.version == "SLv":
    cfg = SLv
train_sets = ('train', args.type)
testset = ('val', args.type)
# train_sets = 'train'
ssd_dim = 300  # only support 300 now
rgb_means = (104, 117, 123)  # only support voc now
num_classes = args.num_classes
if args.fg: num_classes = 2
batch_size = args.batch_size
accum_batch_size = 32
#iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
#stepvalues = (80000, 100000, 120000)
stepvalues = (30000, 50000, 70000)
gamma = 0.1
momentum = 0.9
exp = args.exp_name
os.makedirs('weights'+os.sep+exp, exist_ok=True)
net = build_ssd('train', 300, num_classes)
vgg_weights = torch.load(args.save_folder + args.basenet)
print('Loading base network...')
net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

print('Initializing weights...')
# initialize newly added layers' weights with xavier method
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)
if args.trained_model is not None:
    net.load_state_dict(torch.load(args.trained_model))

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, args.th, True, 0, True, 1, 0.5, False, neg_threshold = args.neg_th)

def test_loss(net, criterion, cuda, epoch):
    #net.eval()
    dataset = SLDetection(SLroot, testset, BaseTransform(
        ssd_dim, rgb_means), SLAnnotationTransform(fg=args.fg))

    
    num_images = len(dataset)
    batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                              shuffle=True, collate_fn=detection_collate))
    epoch_loss = 0
    pbar = trange(num_images//batch_size)
    for i in pbar:
        images, targets = next(batch_iterator)
        print(images.size())
        if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda()) for anno in targets]
        else:
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]
        # forward
        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = args.alpha * loss_l + loss_c
        epoch_loss += loss.data[0]
        pbar.set_description("val loss %f" % (epoch_loss/(i+1)))
    return epoch_loss    

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')
    
    dataset = SLDetection(SLroot, train_sets, BaseTransform(
        ssd_dim, rgb_means), SLAnnotationTransform(fg=args.fg))

    testset = SLDetection(SLroot, train_sets, None, SLAnnotationTransform(fg=args.fg))
    
    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    epoch_loss = 0
    epochs = max_iter // epoch_size
    print("epoch_size", epoch_size)
    print("epochs", epochs)
    steps = 0
    counter = 0
    for epoch in range(epochs):
        pbar = trange(epoch_size)
        # create batch iterator
        shuffle = True
        if batch_size==1: shuffle=False
        batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                              shuffle=shuffle, collate_fn=detection_collate))
        loc_loss = 0
        conf_loss = 0
        epoch_loss = 0
        counter = 0
        print("Epoch starting:", epoch, steps)
        print_learning_rate(optimizer)
        #test_loss(net, criterion, args.cuda, epoch)
        net.train()
        for iteration in pbar:
            steps += 1
            if steps in stepvalues:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
                # reset epoch loss counters
            try:
                # load train data
                images, targets = next(batch_iterator)
                # print(images)
                # print(targets)
                if args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(anno.cuda()) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno) for anno in targets]
                # forward
                out = net(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_c + args.alpha * loss_l 
                loss.backward()
                optimizer.step()
                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]
                epoch_loss += loss.data[0]
                if batch_size==1:
                    image = testset.pull_image(counter)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pos = (criterion.pos_priors * 300).int()
                    ann = Ann(dets=pos.cpu().numpy())
                    AnnImg(None, ann, npimg=rgb_image).plot(label=False).save('tmp/'+str(counter)+'pos.png')
                    AnnImg(None, ann, npimg=rgb_image).plot(label=False, rect=False).save('tmp/'+str(counter)+'posc.png')
                    neg = (criterion.neg_priors * 300).int()
                    ann = Ann(dets=neg.cpu().numpy())
                    AnnImg(None, ann, npimg=rgb_image).plot(label=False).save('tmp/'+str(counter)+'neg.png')
                    AnnImg(None, ann, npimg=rgb_image).plot(label=False, rect=False).save('tmp/'+str(counter)+'negc.png')
                    counter += 1
                    #print(a+b+c)
                pbar.set_description("loss %f" % (epoch_loss/(iteration+1)))
            except RuntimeError as err:
                print("Error", err)
        torch.save(net.state_dict(), 'weights/'+exp+'/ssd300_SL_epoch_' +
                   repr(epoch) + '.pth')
        print('epoch ' + repr(iteration//epoch_size) + ' || Loss: %.4f ||' % (epoch_loss/epoch_size))
        print('epoch ' + repr(iteration//epoch_size) + ' || Conf Loss: %.4f ||' % (conf_loss/epoch_size))
        print('epoch ' + repr(iteration//epoch_size) + ' || Loc Loss: %.4f ||' % (loc_loss/epoch_size))
        #test(epoch, args.cuda)
        #test_loss(net, criterion, args.cuda, epoch)
        
    torch.save(net.state_dict(), args.save_folder + '' + args.version + '.pth')


def print_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        print('lr', param_group['lr'])


    
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
