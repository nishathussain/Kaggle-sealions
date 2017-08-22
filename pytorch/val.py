import cv2
from data import *
from ssd import build_ssd

from torch.autograd import Variable
import torch.utils.data as data
from tqdm import trange
from sealions import SLroot, SL_CLASSES as labelmap

from data import v2, v1, detection_collate, BaseTransform
from sealions import SLroot, SLAnnotationTransform, SLDetection

def add_col2(arr, val, pos=0):
    N = arr.size(0)
    d = arr.size(1)
    #print(N,d,arr,arr[:,:pos], arr[:,pos:])
    ret = torch.ones((N,d+1)).float().cuda() * val
    if pos!=0: 
        ret[:,:pos] = arr[:,:pos]
    if pos!=d: 
        ret[:,pos+1:] = arr[:,pos:]
    return ret

def get_dets(d, iid, x=0, y=0):
    
    #assert d.size(0)==6 and d.size(1)==200 and d.size(2)==5
    ret = []
    for cl in range(1,d.size(0)):
        ret.append(add_col2(d[cl],cl-1))
        
    ret =  torch.cat(ret)
    return addxy(ret, iid, x, y)

def addxy(ret, iid, x, y):
    ret[:,2:] *= 300
    ret[:,2] += x
    ret[:,3] += y
    ret[:,4] += x
    ret[:,5] += y
    ret[:,1] *= 100
    ret = add_col2(ret, iid)
    return ret.int()

def get_iidxy(iidxy):
    _,iid,x_y = iidxy.split('/')
    x,y = x_y.split('_')
    return int(iid),int(x),int(y)


def preds_targets(net, dataset, cuda, num_images):
    batch_iterator = iter(data.DataLoader(dataset, 1,
                                              shuffle=False, collate_fn=detection_collate))
    
    # dump predictions and assoc. ground truth to text file for now
    #num_images = len(dataset)
    targs = []
    dets = []
    pbar = trange(num_images)
    total = 0
    for i in pbar:
        #print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        iid,xx,yy = get_iidxy(dataset.ids[i])
        images, targets = next(batch_iterator)
        #print(targets)
        if len(targets[0])!=0:
            t = torch.cat(targets.copy(),0)
            t = torch.cat([t[:,4:],t[:,:4]],1)
            t = add_col2(t,1.0,1)
            t = addxy(t,iid,xx,yy).int()
            targs.append(t)
        #print(t)
        x = Variable(images)
        if cuda:
            x = x.cuda()
        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        d = get_dets(detections[0],iid,xx,yy)
        dets.append(d)
        #d = d.numpy()
        total += len(torch.nonzero(d[:,2]>50))
        pbar.set_description("counts %d" % total)
    dets = torch.cat(dets)
    targs = torch.cat(targs)
    return dets, targs


def preds_targets_batch(net, dataset, cuda, num_images, batch_size):
    batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                              shuffle=False, collate_fn=detection_collate))
    
    # dump predictions and assoc. ground truth to text file for now
    #num_images = len(dataset)
    targs = []
    dets = []
    for i in trange(num_images):
        #print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        iid,xx,yy = get_iidxy(dataset.ids[i])
        images, targets = next(batch_iterator)
        #print(targets)
        if len(targets[0])!=0:
            t = torch.cat(targets.copy(),0)
            t = torch.cat([t[:,4:],t[:,:4]],1)
            t = add_col2(t,1.0,1)
            t = addxy(t,iid,xx,yy).int()
            targs.append(t)
        #print(t)
        x = Variable(images)
        if cuda:
            x = x.cuda()
        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        dets.append(get_dets(detections[0],iid,xx,yy))
    dets = torch.cat(dets)
    targs = torch.cat(targs)
    return dets, targs


if __name__ == "__main__":
    num_classes = 2
    exp = 'th10to50_resume8/ssd300_SL_epoch_17'
    epoch = 0
    ssd_dim = 300  # only support 300 now
    rgb_means = (104, 117, 123)  # only support voc now
    name = 'non_zero_ann'
    #name = 'zero'
    ds = 'val'
    testset = (ds, name)
    net = build_ssd('test', 300, num_classes)    # initialize SSD
    #trained_model = 'weights/ssd300_SL_epoch_'+  repr(epoch) + '.pth'
    trained_model = 'weights/'+exp+'.pth'
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    dataset = SLDetection(SLroot, testset, BaseTransform(
        ssd_dim, rgb_means), SLAnnotationTransform())

    num_images = len(dataset)
    preds, targets = preds_targets(net, dataset, True, num_images)
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    np.savetxt(exp+"_"+ds+"_"+name+"_preds.csv", p[p[:,2]>1], delimiter=",", fmt="%d")
    np.savetxt(exp+"_"+ds+"_"+name+"_truth.csv", t, delimiter=",", fmt="%d")
