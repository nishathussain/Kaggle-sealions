from random import seed, shuffle
import os
import os.path
import sys
from tqdm import tqdm
from polarbear.aimg import AnnImg
from polarbear.ann import Ann
import numpy as np
from random import shuffle,seed

class AnnImgs(object):
    
    #aimgs, ids, 
    def __init__(self, iids, img_path, ann_path, scale_file=None):
        self.scale_file = scale_file
        self.iids = iids
        self.img_path = img_path
        self.ann_path = ann_path
        self.scales = {}
        if scale_file is not None: 
            self.scales = { x[0]:x[1] for x in np.loadtxt(scale_file, delimiter=",",
                                     skiprows=1).astype('int32')}

    def shuffle(self, s=0):
        self.iids = sorted(self.iids)
        seed(0)
        shuffle(self.iids)
        return self
    
    def take(self, end=1000, start=0):
        self.iids = self.iids[start:end]        
        return self

    def tile(self, d=300, s=200):
        for aimg, iid in self:
            for timg,x,y in aimg.tile(d,s):
                yield timg, iid, x, y
        
    def aimg(self, iid):
        img_path=self.img_path.format(iid=iid)
        ann_path=self.ann_path.format(iid=iid)
        scale=self.scales.get(iid,None)
        #if scale==0: scale=40
        if not os.path.exists(ann_path):
            ann_path = None
            return AnnImg(img_path=img_path), iid
        else:
            return AnnImg(img_path=img_path, ann_path=ann_path, scale=scale), iid
    
    def ann(self, iid):
        ann_path=self.ann_path.format(iid=iid)
        scale=self.scales.get(iid,None)
        if not os.path.exists(ann_path):
            ann_path = None
            return None, iid
        else:
            return Ann(file=ann_path, scale=scale), iid
    
    def anns(self):
        for iid in self.iids:
            yield self.ann(iid)
    
    def posneg(self, d):
        X,Y = [],[]
        for aimg,_ in self:
            x,y,_,_ = aimg.posneg(d)
            X.append(x)
            Y.append(y)
        return np.concatenate(X), np.concatenate(X)
        
    
    def __iter__(self):
        for iid in tqdm(self.iids):
            yield  self.aimg(iid)
            
    def __getitem__(self, i):
        return self.aimg(self.iids[i])
    
    def aimgs():
        for iid in self.iids:
            yield self.aimg(iid)
            
    def mkpath(self, path):
        path = os.path.dirname(os.path.abspath(path))
        os.makedirs(path, exist_ok=True)
        
    def size(self): return len(self.iids)        
    
    def split(self, ratio=0.2):
        all_ids = self.iids
        seed(0)
        shuffle(all_ids)
        N = len(all_ids)
        Nv = int(ratio*N)
        train_ids, val_ids = all_ids[:-Nv], all_ids[-Nv:]
        train_ids = sorted([int(f) for f in train_ids] )
        val_ids = sorted([int(f) for f in val_ids] )
        return AnnImgs(train_ids, self.img_path, self.ann_path, self.scale_file), \
                AnnImgs(val_ids, self.img_path, self.ann_path, self.scale_file)
    
    def savedir(self, aimg_dir, prep=None):    
        img_dir = aimg_dir + os.sep + "Images"
        ann_dir = aimg_dir + os.sep + "Annotations"
        self.savedirs(img_dir, ann_dir, prep)
    
    def savedirs(self, img_dir, ann_dir, prep=None):
        img_path = img_dir + os.sep + "{iid}.jpg"
        ann_path = ann_dir + os.sep +"{iid}.csv"
        self.save(img_path, ann_path, prep)
    
    def save(self, img_path, ann_path, prep=None):
        self.mkpath(img_path)
        self.mkpath(ann_path)
        for aimg,iid in self:
            ipath = img_path.format(iid=iid)
            apath = ann_path.format(iid=iid)
            if prep is not None: aimg = prep(aimg)
            aimg.save(ipath, apath)
    
    def save_tiles_dir(self, aimg_dir, d=300, s=300, ann_file="xml", plot=False, rect=True, label=True):
        ipath = aimg_dir + os.sep + "Images" + os.sep + \
                "{iid}"+ os.sep+"{x}_{y}.jpg"
        apath = aimg_dir + os.sep + "Annotations" + os.sep +\
                "{iid}"+ os.sep+"{x}_{y}.xml"
        ppath = aimg_dir + os.sep + "Plot" + os.sep +\
                "{iid}"+ os.sep+"{x}_{y}.jpg"
        for timg, iid, x, y in self.tile(d=d,s=s):
            img_path = ipath.format(iid=iid,x=x,y=y)
            ann_path = apath.format(iid=iid,x=x,y=y)
            plot_path = ppath.format(iid=iid,x=x,y=y)
            self.mkpath(img_path)
            #self.mkpath(ann_path)
            #    self.mkpath(plot_path)
            #print(img_path,ann_path)
            timg.save(img_path)
            #timg.plot(rect=rect, label=label).save(plot_path)
            #break
            
    def save_plot(self, plot_path):
        for aimg,iid in tqdm(self):
            filepath = plot_path.format(iid=iid)
            #print(filepath)
            aimg.plot().save(filepath)
            filepath = plot_path.format(iid=str(iid)+"c")
            aimg.plot(rect=False, label=False).save(filepath)
            
    def counts(self, sortby=-2, reverse=True, take=100):
        ret = []
        for aimg,iid in self:
            ret.append([iid] + aimg.counts.tolist() + [aimg.fpups().count,aimg.count] )
        arr = np.array(ret)
        if sortby is not None: arr = arr[np.argsort(arr[:,sortby])]
        if reverse: arr = arr[::-1]
        if take is not None: arr = arr[:take]
        return arr
    
class AnnImgsDirs(AnnImgs):
    
    #aimgs, ids, 
    def __init__(self, img_dir, ann_dir, sort=True, sortint=True, scale_file=None, ann_ext=".csv", ann_chk=False):
        files = os.listdir(img_dir)
        if ann_chk: files = os.listdir(ann_dir)
        if sortint: iids = sorted([int(os.path.splitext(f)[0]) for f in files] )
        else: iids = sorted([(os.path.splitext(f)[0]) for f in files] )
        img_path = img_dir + os.sep + "{iid}.jpg"
        ann_path = ann_dir + os.sep +"{iid}"+ann_ext
        AnnImgs.__init__(self, iids, img_path, ann_path, scale_file)
        
    
class AnnImgsDir(AnnImgsDirs):
    
    #aimgs, ids, 
    def __init__(self, aimg_dir, sort=True, sortint=True, ann_ext=".csv", test=None, ann_chk=False):
        img_dir = aimg_dir + os.sep + "Images"
        ann_dir = aimg_dir + os.sep + "Annotations"
        if test is not None: ann_dir = aimg_dir + os.sep + "Annotations_" + test
        scale_file = aimg_dir + os.sep + "scale.csv"
        if not os.path.exists(scale_file): scale_file=None
        else: print("loading scale file", scale_file)
        AnnImgsDirs.__init__(self, img_dir, ann_dir, scale_file=scale_file, sortint=sortint, ann_ext=ann_ext, ann_chk=ann_chk)
        
class AnnImgsTilesDir(AnnImgs):
    
    #aimgs, ids, 
    def __init__(self, aimg_dir, sort=True, sortint=True, scale=True):
        img_dir = aimg_dir + os.sep + "Images"
        ann_dir = aimg_dir + os.sep + "Annotations"
        
        files = os.listdir(img_dir)
        iids = sorted([int(os.path.splitext(f)[0]) for f in files] )
        
        all_files = []
        for iid in iids:
            files = os.listdir(img_dir + os.sep + str(iid))
            files = [str(iid) + os.sep + os.path.splitext(f)[0] for f in sorted(files)]
            all_files = all_files + (files)
            
        AnnImgs.__init__(self, all_files, img_dir + os.sep + "{iid}.jpg", ann_dir + os.sep + "{iid}.xml")
       
              
        
            