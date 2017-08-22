from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import sys
import os
import time
import argparse
import numpy as np
import pickle
from collections import Counter, defaultdict

from utils.box_utils import jaccard, point_form
from data import AnnotationTransform, VOCDetection, base_transform
from data import VOCroot
from data import VOC_CLASSES as labelmap
from ssd import build_ssd
import torch.utils.data as data
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/',
                    type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6,
                    type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def voc_ap(rec, prec):
    """VOC_AP average precision calculations using 11-recall-point based AP
    metric (VOC2007)
    [precision integrated to recall]
    Params:
        rec (FloatTensor): recall cumsum
        prec (FloatTensor): precision cumsum
    Return:
        average precision (float)
    """
    ap = 0.
    for threshold in np.arange(0., 1., 0.1):
        if torch.sum((rec >= threshold)) == 0:  # if no recs are >= this thresh
            p = 0
        else:
            # largest prec where rec >= thresh
            p = torch.max(prec[rec >= threshold])
        ap += p / 11.
    return ap


def eval_net(net, cuda, dataset, transform, top_k):
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(dataset)
    ovthresh = 0.5
    confidence_threshold = 0.01
    num_classes = 0

    # per class

    fp = defaultdict(list)
    tp = defaultdict(list)
    gts = defaultdict(list)
    precision = Counter()
    recall = Counter()
    ap = Counter()

    for i in range(num_images//100):
        if i % 10 == 0:
            print('Evaluating image {:d}/{:d}....'.format(i + 1, num_images))
        t1 = time.time()
        img = dataset.pull_image(i)
        img_id, anno = dataset.pull_anno(i)
        anno = torch.Tensor(anno).long()
        x = cv2.resize(np.array(img), (300, 300)).astype(np.float32)
        x -= (104, 117, 123)
        x = x.transpose(2, 0, 1)
        x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.size[0], img.size[1],
                              img.size[0], img.size[1]])
        # for each class
        if num_classes == 0:
            num_classes = detections.size(1)
        for cl in range(1, detections.size(1)):
            dets = detections[0, cl, :]
            mask = dets[:, 0].ge(confidence_threshold).expand(
                5, dets.size(0)).t()
            # all dets w > 0.01 conf for class
            dets = torch.masked_select(dets, mask).view(-1, 5)
            mask = anno[:, 4].eq(cl-1).expand(5, anno.size(0)).t()
            # all gts for class
            truths = torch.masked_select(anno, mask).view(-1, 5)
            if truths.numel() > 0:
                # there exist gt of this class in the image
                # check for tp & fp
                truths = truths[:, :-1]
                if dets.numel() < 1:
                    fp[cl].extend([0] * truths.size(0))
                    tp[cl].extend([0] * truths.size(0))
                    gts[cl].extend([1] * truths.size(0))
                    # gts[cl][-1] += truths.size(0)
                    continue
                preds = dets[:, 1:]
                # compute overlaps
                overlaps = jaccard(truths.float() /
                                   scale.unsqueeze(0).expand_as(truths), preds)
                # found = if each gt obj is found yet
                found = [False] * overlaps.size(0)
                maxes, max_ids = overlaps.max(0)
                maxes.squeeze_(0), max_ids.squeeze_(0)
                for pb in range(overlaps.size(1)):
                    max_overlap = maxes[pb]
                    gt = max_ids[pb]
                    if max_overlap > ovthresh:  # 0.5
                        if found[gt]:
                            # duplicate
                            fp[cl].append(1)
                            tp[cl].append(0)
                            gts[cl].append(0)  # tp
                        else:
                            # not yet found
                            tp[cl].append(1)
                            fp[cl].append(0)
                            found[gt] = True  # mark gt as found
                            gts[cl].append(1)  # tp
                    else:
                        fp[cl].append(1)
                        tp[cl].append(0)
                        gts[cl].append(0)  # tp
            else:
                # there are no gts of this class in the image
                # all dets > 0.01 are fp
                if dets.numel() > 0:
                    fp[cl].extend([1] * dets.size(0))
                    tp[cl].extend([0] * dets.size(0))
                    gts[cl].extend([0] * dets.size(0))
        if i % 10 == 0:
            print('Timer: %.4f' % (time.time()-t1))
    for cl in range(1, num_classes):
        if len(gts[cl]) < 1:
            continue
        # for each class calc rec, prec, ap
        tp_cumsum = torch.cumsum(torch.Tensor(tp[cl]), 0)
        fp_cumsum = torch.cumsum(torch.Tensor(fp[cl]), 0)
        gt_cumsum = torch.cumsum(torch.Tensor(gts[cl]), 0)
        rec_cumsum = tp_cumsum.float() / gt_cumsum[-1]
        prec_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)
        ap[cl] = voc_ap(rec_cumsum, prec_cumsum)
        recall[cl] = rec_cumsum[-1]
        precision[cl] = prec_cumsum[-1]
        print('class %d rec %.4f prec %.4f AP %.4f tp %.4f fp %.4f, \
        gt %.4f' % (cl, recall[cl], precision[cl], ap[cl], sum(tp[cl]),
              sum(fp[cl]), sum(gts[cl])))
    # mAP = mean of APs for all classes
    mAP = sum(ap.values()) / len(ap)
    print('mAP', mAP)
    return mAP


if __name__ == '__main__':
    # load net
    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(VOCroot, 'test', None, AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    eval_net(net, args.cuda, dataset, base_transform(
        net.size, (104, 117, 123)), args.top_k)
