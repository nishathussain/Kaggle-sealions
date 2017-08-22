from layers.box_utils import nms
import torch
import os
from polarbear import *
from layers.box_utils import nms, match, jaccard
import torch

def getBoxes(preds, iid, conf=90, gth=0.5):
    preds1= preds[preds[:,0]==iid][:,1:]
    #print(preds1)
    tmp = preds1[preds1[:,1]>conf]
    p90 = torch.from_numpy(tmp)
    if(p90.nelement()>0):
        ids,cnt = nms(p90[:,2:].float(), p90[:,1].float(), gth, 200)
        ann1 = torch.index_select(p90,0,ids[:cnt].cpu().long())
        return ann1
    else: return None

def get_preds(preds, iid):
    return preds[preds[:,0]==iid][:,1:]
    
def classNMS(ann1, th=0.1):
    ret = []
    for i in range(5):
        mask = torch.nonzero(ann1[:,0]==i)
        #print(mask)
        if(mask.nelement()>0):
            clann = ann1.index_select(0, mask[:,0])
            idx, cnt = nms(clann[:,2:].float(), clann[:,1].float(), th, 200)
            #(cnt, clann.size(0))
            tmp = clann.index_select(0, idx[:cnt].cpu())
            if tmp.nelement()>0: ret.append(tmp)
    return torch.cat(ret)

def plots(ds="train", name="non_zero_ann"):
    
    preds = np.loadtxt(ds+"_"+name+"_preds.csv",  delimiter=",").astype('int32')
    val = DataSource().dataset(ds)

    model = "th90"
    os.makedirs(ds+"plots", exist_ok=True)
    for iid in tqdm(val.iids):
        aimg,_ = val.aimg(iid)
        aimg.plot().save('plots/'+th+str(iid)+'.jpg')
        ann1 = getBoxes(preds, iid, conf=50, gth=1.0)
        #print(ann1)
        if ann1 is not None:
            aimg.plot(Ann(dets=ann1.numpy())).save(model_ds+'plots/'+th+str(iid)+'_50_1.0.jpg')
        ann1 = getBoxes(preds, iid, conf=90, gth=0.5)
        if ann1 is not None:
            aimg.plot(Ann(dets=ann1.numpy())).save(model_ds+'plots/'+th+str(iid)+'_90_0.5.jpg')
        if ann1 is not None:
            ann1 = classNMS(ann1, th=0.1)
            #if ann1 is not None:
            aimg.plot(Ann(dets=ann1.numpy())).save(model_ds+'plots/'+th+str(iid)+'_90_0.5_0.1.jpg')

def plots2(ds="val", name="non_zero_ann", model="th90"):
    preds = np.loadtxt("../data/exp/"+ds+"_"+name+"_preds.csv",  delimiter=",").astype('int32')
    val = DataSource().dataset(ds)
    prefix = ds+"plots"+model+"/"
    os.makedirs(prefix, exist_ok=True)

    for iid in tqdm(val.iids):
            aimg,_ = val.aimg(iid)
            aimg.plot().save(prefix+str(iid)+'.jpg')
            ann1 = getBoxes(preds, iid, conf=50, gth=1.0)
            #print(ann1)
            if ann1 is not None:
                aimg.plot(Ann(dets=ann1.numpy())).save(prefix+str(iid)+'_50_1.0.jpg')
            
def gen_preds(ds="val", name="non_zero_ann"):
    preds = np.loadtxt("../data/exp/"+ds+"_"+name+"_preds.csv",  delimiter=",").astype('int32')
    val = DataSource().dataset(ds)
    
    for iid in (val.iids):
            aimg,_ = val.aimg(iid)
            ann1 = getBoxes(preds, iid, conf=50, gth=1.0)
            yield aimg, AnnImg(aimg.img, Ann(dets=ann1.numpy()))

            
def gen_anns(ds, file, conf=50, gth=1.0):
    preds = np.loadtxt(file,  delimiter=",").astype('int32')
    val = DataSource().dataset(ds)
    
    for iid in (val.iids):
            gt_ann,_ = val.ann(iid)
            pd_dets = getBoxes(preds, iid, conf=conf, gth=gth)
            if pd_dets is None:
                yield gt_ann, None, iid
            else: yield gt_ann, Ann(dets=pd_dets.numpy()), iid
            
def prec_recall(gt_dets, pd_dets, th=0.5, filter_pups=False):
    if filter_pups: 
        #print("gt_dets",gt_dets.shape)
        gt_dets = gt_dets[gt_dets[:,0]!=4] 
        #print("gt_dets",gt_dets.shape)
        pd_dets = pd_dets[pd_dets[:,0]!=4] 
    bgt = torch.from_numpy(gt_dets).float()
    bpd = torch.from_numpy(pd_dets).float()
    overlaps = jaccard(bgt[:,2:], bpd[:,2:])
    overlaps[overlaps<th] = 0
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    best_prior_idx = best_prior_idx.squeeze()
    

    tp_gt = torch.nonzero(best_prior_idx)
    if tp_gt.nelement() == 0:
        return 0, 0, ([], [], bgt, bpd)
    tp_gt_boxes = bgt.index_select(0,tp_gt.squeeze())
    
    tp_pd = torch.index_select(best_prior_idx, 0, tp_gt.squeeze())
    tp_pd_boxes = bpd.index_select(0,tp_pd.squeeze())
    
    fn = torch.nonzero(best_prior_idx==0)
    if fn.nelement() == 0:
        fn_boxes = []
    else :fn_boxes = bgt.index_select(0,fn.squeeze())
    
    fp = torch.ones((len(bpd))).cpu()
    fp = fp.index_fill_(0,tp_pd,0).nonzero()
    if fp.nelement() == 0:
        fp_boxes = []
    else: fp_boxes = bpd.index_select(0,fp.squeeze())
    prec = len(tp_pd)/(len(tp_pd)+len(fp))
    recall = len(tp_gt)/(len(tp_gt)+len(fn))
    #print("%.2f" % prec, "%.2f" % recall, len(tp_gt_boxes), len(fn_boxes), len(fp_boxes))
    return prec, recall, (tp_gt_boxes, tp_pd_boxes, fn_boxes, fp_boxes)

def prec_recall_all(ds, file, conf, gth, pall=False, th=0.1, filter_pups=False):
    tp,fp,fn = 0,0,0
    for gt_ann, pd_ann, iid in gen_anns(ds, file, conf=conf, gth=gth):
        if pd_ann is not None:
            prec, recall, (tp_gt_boxes, tp_pd_boxes, fn_boxes, fp_boxes) = prec_recall(gt_ann.dets, pd_ann.dets, th=th, filter_pups=filter_pups)
            if pall: print(iid, "%.2f" % prec, "%.2f" % recall, len(tp_pd_boxes), len(fp_boxes), len(fn_boxes))
            tp += len(tp_pd_boxes)        
            fp += len(fp_boxes)        
            fn += len(fn_boxes)   
        else: fn +=  gt_ann.count
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    return prec, recall
    
