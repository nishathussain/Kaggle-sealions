import os
import os.path
import sys
from PIL import Image
import numpy as np
from polarbear.aimg import AnnImg
from polarbear.ann import Ann
from random import seed, shuffle
from tqdm import tqdm
import pandas as pd

CLASS_NAMES = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups')
        
CLASS_COLORS = (
            (255,0,0, 128),          # red
            (250,10,250,128),       # magenta
            (84,42,0,128),          # brown 
            (30,60,180,128),        # blue
            (35,180,20,128),        # green
            )

SOURCEDIR = '/home/nishat/deepML/polarbear/data/'

class SeaLions(object):
    
    def __init__(self, dataset="all", iids=None, sourcedir=SOURCEDIR):
        
        self.sourcedir = sourcedir
      
        self.classes = 5
        self.dataset = dataset
        self.source_paths = {
            'images': [sourcedir, '{dataset}', 'Images', '{iid}.jpg'],
            'anns':   [sourcedir, '{dataset}', 'Annotations', '{iid}.csv'],
            'dotted':   [sourcedir, '{dataset}', 'Dotted', '{iid}.jpg'],
            'tiles':   [sourcedir, '{dataset}', 'Tiles_{d}_{s}', '{iid}','Images','{cnt}.jpg'],
            'tiles':   [sourcedir, '{dataset}', 'TilesBinary_{d}_{s}', '{label}','{cnt}.jpg'],
            'tiles_ann':   [sourcedir, '{dataset}', 'Tiles_{d}_{s}', '{iid}','Annotations','{cnt}.csv'],
            'binary':   [sourcedir, '{dataset}', 'binary', '{iid}.jpg'],
            'masked':   [sourcedir, '{dataset}', 'masked', '{iid}.jpg'],
            'masked_tiles':   [sourcedir, '{dataset}', 'masked_tiles_link', '{iid}','{x}_{y}.jpg'],
            'ssd_score':   [sourcedir, 'lmdb','{dataset}_{x}_{y}','comp4_det_test_{cname}.txt'],
            'scale':   [sourcedir, 'sealion_scale.csv'],
            'plot':   [sourcedir, '{dataset}', 'plot', '{iid}_{type}.jpg']
            }
        if iids is None:
            self.iids = self.get_iids()
        else: self.iids = iids

    def mask_tiles(self):
        sl = self
        for iid in tqdm(sl.iids):
            masked_path = sl.path('masked',iid=iid)
            image_path = sl.path('images',iid=iid)
            sl.mkpath('masked_tiles',iid=iid)
            img = AnnImg(Image.open(image_path))
            for aimg,x,y in AnnImg(Image.open(masked_path)).genCropXY(300,200):
                if(aimg.np().sum()!=0):
                    aimg = img.crop_xyd(x,y,300)
                    tile_path = sl.path('masked_tiles', iid=iid, x=x, y=y)
                    aimg.save(tile_path)
    
    def getScore(self,x=0,y=1000, iid=None):
        if iid is not None:
            x=(iid//1000)*1000
            y=x+1000
        ret = []
        for cl in range(5):
            filepath = self.path("ssd_score",cname=CLASS_NAMES[cl],x=x,y=y)
            df = pd.read_csv(filepath, delimiter=" ", header=-1)
            arr = df.as_matrix()
            ret.append(self.getbbox(cl,arr))
        ret = np.concatenate(ret,axis=0)
        if iid is not None: 
            return ret[ret[:,0]==iid]
        return ret
        
    
    def getCounts(self):
        counts = np.loadtxt( "../train.csv",delimiter=",",skiprows=1)
        return counts.astype('int32')[:,1:]

    def getScale(self, iid):
        #arr = np.loadtxt("../sealion_scale.csv",delimiter=",",skiprows=1).astype('int32')
        arr = np.loadtxt(self.path('scale'),delimiter=",",skiprows=1).astype('int32')
        ratios = [1.0,0.8,0.7,0.6,0.5]
        d = np.array([arr[iid][1]*r for r in ratios])
        #d = {x[0]:x[1]*ratios for x in arr}
        ann = self.aimg(iid).ann
        #scale = d[iid]
        scales = d[ann[:,0]]
        dets = [ann[:,0], ann[:,0]*0+100, ann[:,1]-scales, ann[:,2]-scales, ann[:,1]+scales, ann[:,2]+scales]
        return np.vstack(dets).astype('int32').transpose()
        
    def getbbox(self, cl, arr):
        ret = []
        for fname, conf, xmin, ymin, xmax, ymax in arr:
            iid = int(fname.split("/")[-2])
            x,y = fname.split("/")[-1][:-4].split("_")
            x,y = int(x),int(y)
            xmin, ymin, xmax, ymax  = xmin+x, ymin+y, xmax+x, ymax+y
            ret.append([iid,cl,int(conf*100),xmin,ymin,xmax,ymax])
        return np.array(ret)
    
    def __iter__(self):
        for iid in tqdm(self.iids):
            yield iid, self.aimg(iid)
            
    def __getitem__(self, i):
        return self.iids[i], self.aimg(self.iids[i])
        
    def tile(self, d, s):
        for iid in tqdm(self.iids):
            self.mkpath('tiles',iid=iid, d=d, s=s)
            self.mkpath('tiles_ann',iid=iid, d=d, s=s)
            cnt = 0
            for aimg in self.aimg(iid).genCrop(d,s):
                
                img_path = self.path('tiles', d=d, s=s, iid=iid, cnt=cnt)
                ann_path = self.path('tiles_ann', d=d, s=s, iid=iid, cnt=cnt)
                aimg.save(img_path, ann_path)
                cnt += 1
    
    def tile_binary(self, d, s):
        self.mkpath('tiles',label="yes", d=d, s=s)
        self.mkpath('tiles',label="no", d=d, s=s)
        cnts = {"yes":0, "no":0}
        for iid in tqdm(self.iids):
            
            for aimg in self.aimg(iid).genCrop(d,s):
                if(aimg.count>0): label="yes"
                else: 
                    label="no"
                    if cnts["no"]>cnts["yes"]: continue
                img_path = self.path('tiles', d=d, s=s, label=label, cnt=cnts[label])
                aimg.save(img_path)
                cnts[label] += 1
    
    def tile_binary_neg(self, d, s, model, batch_size=32):
        ret = []
        batch = {}
        for iid in tqdm(self.iids):
            for aimg in self.aimg(iid).genCrop(d,s):
                if(aimg.count==0): 
                    batch.append(aimg.np())
                    if(len(batch)==batch_size):
                        X = np.array(batch)
                        batch = {}
                        Y = model.predict(X)
                        mask = Y[:,0]>prob
                        ret.append(X[mask])
            break
        return ret
                        
    def path(self, name, **kwargs):
        """Return path to various source files"""
        if not "dataset" in kwargs: kwargs["dataset"] = self.dataset
        path = os.path.join(*self.source_paths[name]).format(**kwargs)
        return path        
    
    def dirpath(self, name, **kwargs):
        if not "dataset" in kwargs: kwargs["dataset"] = self.dataset
        path = os.path.join(*self.source_paths[name][:-1]).format(**kwargs)
        return path
    
    def mkpath(self, name, **kwargs):
        path = self.dirpath(name, **kwargs)
        os.makedirs(path, exist_ok=True)
        
    def get_iids(self):
        files = os.listdir(self.dirpath('images'))
        return sorted([int(os.path.splitext(f)[0]) for f in files] )
        
    def aimgs(self):
        for iid in self.iids:
            yield self.aimg(iid)
            
    def aimg(self, iid):
            img_path = self.path('images', iid=iid)
            ann_path = self.path('anns', iid=iid)
            if not os.path.exists(ann_path):
                ann_path = None
            return AnnImg(img_path=img_path, ann_path=ann_path)
        
    def save(self, ds):
        self.mkpath("images",  dataset=ds)
        self.mkpath("anns",  dataset=ds)
            
        for iid in tqdm(self.iids):
            img_path = self.path("images", iid=iid, dataset=ds)
            ann_path = self.path("anns", iid=iid, dataset=ds)
            self.aimg(iid).save(img_path, ann_path)
            
    def plot(self):
        self.mkpath("dotted")
        for iid in tqdm(self.iids):
            fpath = self.path("dotted",iid=iid)
            self.aimg(iid).plot().save(fpath)
            
    def val_split(self, ratio=0.2):
        all_ids = self.iids
        seed(0)
        shuffle(all_ids)
        N = len(all_ids)
        Nv = int(ratio*N)
        train_ids, val_ids = all_ids[:-Nv], all_ids[-Nv:]
        SeaLions(iids=train_ids).save("train")
        SeaLions(iids=val_ids).save("val")
        
      
 