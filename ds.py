import os
import os.path
import sys
from PIL import Image
import numpy as np
from polarbear.aimg import AnnImg
from polarbear.ann import Ann
from polarbear.aimgs import *
from random import seed, shuffle
from tqdm import tqdm
import pandas as pd
from scipy.misc import imsave, imread

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

class DataSource(object):
    
    def __init__(self, name="correct", dstype="datasource", sourcedir=SOURCEDIR):
        
        self.sourcedir = sourcedir + os.sep + name
        
        self.source_paths = {
            'dataset': [self.sourcedir, '{dataset}'],
            'images': [self.sourcedir, '{dataset}', 'Images', '{iid}.jpg'],
            'anns':   [self.sourcedir, '{dataset}', 'Annotations', '{iid}.csv'],
            'anns_test':   [self.sourcedir, '{dataset}', 'Annotations_{test}', '{iid}.csv'],
            'plot_test':   [self.sourcedir, '{dataset}', 'Plot_{test}', '{iid}.jpg'],
            'scale':   [self.sourcedir, '{dataset}', 'scale.csv'],
            'plot':   [self.sourcedir, '{dataset}', 'plot', '{iid}_{type}.jpg'],
            'list_file':   [self.sourcedir, '{dataset}','lmdb','{type}_list.txt'],
            'name_file':   [self.sourcedir, '{dataset}','lmdb','{type}_size.txt'],
            'plot_ds':   [self.sourcedir, '{dataset}','plot'],
            'patch':   [self.sourcedir, '{dataset}','patch_{exp}', '{cls}', '{cnt}.jpg'],
            'fpatch':   [self.sourcedir, '{dataset}','fpatch', '{cls}', '{cnt}.jpg'],
            'img_file':   ['Images', '{iid}.jpg'],
            'ann_file':   ['Annotations', '{iid}.xml'],
            'patch_output':   [self.sourcedir,  '{dataset}', 'patch_output', '{iid}','{x}_{y}_{type}.jpg'],
            'mask':   [self.sourcedir,  '{dataset}', 'binary', '{iid}.jpg'],
            'th':   [self.sourcedir,  '{dataset}', 'th', '{iid}','{x}_{y}.th'],
            'ssd_score':   [self.sourcedir, '{dataset}', 'lmdb','{type}','output', 'comp4_det_test_{cname}.txt'],
            'negs':   [self.sourcedir, '{dataset}', 'NegAnnotations', '{iid}.csv'],
            'sub':   [self.sourcedir, '{dataset}', 'sub', '{sub}.csv'],
            
            }
        self.type = dstype
    
    def true_counts(self):
        fpath = self.sourcedir + os.sep + 'train.csv'
        counts = np.loadtxt(fpath, delimiter=',', skiprows=1)
        return counts
        
        
    def create_lmdb(self, dataset, name="default", \
                    non_zero_ann=False, bgclass=False, mask=False, balanced=False):
        self.mkpath("list_file",dataset=dataset)
        f = open(self.path("list_file",dataset=dataset,type=name), 'w')
        ff = open(self.path("name_file", dataset=dataset,type=name), 'w')
        cnt = 0
        bal = 0
        for aimg, iid in self.dataset(dataset):
            ipath = self.path('img_file', iid=iid)
            apath = self.path('ann_file', iid=iid)
            if mask and aimg.np().sum()==0: continue
            if aimg.ann.count==0:
                if balanced and bal<=0:
                    continue
                if non_zero_ann: continue
                else: apath = "Annotations.xml"
                if bgclass: apath = "Annotations_bg.xml"
                bal -= 1
            else: bal += 1
            #print(name)
            s = ipath+" "+apath
            cnt = cnt + 1
            f.write(s+"\n")
            ff.write(ipath+" 300 300\n")
        print("writing "+str(cnt)+" files")
        f.close()
        ff.close()
    
    def create_file(self, dataset, name="default", zero_ann=False):
        self.mkpath("list_file",dataset=dataset)
        f = open(self.path("list_file",dataset=dataset,type=name), 'w')
        ff = open(self.path("name_file", dataset=dataset,type=name), 'w')
        cnt = 0
        bal = 0
        for anns, iid in self.dataset(dataset).anns():
            ipath = self.path('img_file', iid=iid)
            apath = self.path('ann_file', iid=iid)
            #print(name)
            if zero_ann or anns.count>0:
                s = ipath+" "+apath
                cnt = cnt + 1
                f.write(s+"\n")
                ff.write(ipath+" 300 300\n")
        print("writing "+str(cnt)+" files")
        f.close()
        ff.close()
        
    
    def cposneg(self, dataset, d=None, fpups=False, oneclass=False, exp="classes"):
        for i in range(6):
            self.mkpath("patch", dataset=dataset, cls=i, exp=exp)
            
        cnt = 0
        for aimg,iid in self.dataset(dataset):
            #print("iid", iid)
            if fpups: aimg = aimg.fpups()
            if oneclass: aimg = aimg.oneclass()
            for cl,ai in aimg.crop_posneg(d):
                filepath = self.path("patch", dataset=dataset, cls=(cl+1), cnt=cnt, exp=exp)
                ai.save(filepath)
                cnt = cnt + 1
    
    def siamese23(self, dataset, d, exp="s23"):
        self.mkpath("patch", dataset=dataset, cls="True", exp=exp)
        self.mkpath("patch", dataset=dataset, cls="False", exp=exp)
            
        cnt = 0
        for aimg,iid in self.dataset(dataset):
            #print("iid", iid)
            for x,y,cl in aimg.siamese23(d):
                filepath = self.path("patch", dataset=dataset, cls=cl, cnt=cnt, exp=exp)
                x.mergeW(y).save(filepath)
                cnt = cnt + 1
                     
    def posneg(self, dataset, d, mid=False, fpups=False, oneclass=False, exp="", classes=6):
        for i in range(classes):
            self.mkpath("patch", dataset=dataset, cls=i, exp=exp)
        self.mkpath("patch", dataset=dataset, cls=1, exp=exp)
        print(self.path("patch", dataset=dataset, cls=1, cnt=0, exp=exp))
        cnt = 0
        for aimg,iid in self.dataset(dataset):
            #print("iid", iid)
            if fpups: aimg = aimg.fpups()
            if oneclass: aimg = aimg.oneclass()
            X,Y,_,_ = aimg.posneg(d)
            for i in range(X.shape[0]):
                filepath = self.path("patch", dataset=dataset, cls=int(Y[i]), cnt=cnt, exp=exp)
                imsave(filepath, np.transpose(X[i],[1,2,0]))
                cnt = cnt + 1
            if mid:
                X,Y,_,_ = aimg.posmid(d)
                for i in range(X.shape[0]):
                    filepath = self.path("patch", dataset=dataset, cls=int(Y[i]), cnt=cnt, exp=exp)
                    imsave(filepath, np.transpose(X[i],[1,2,0]))
                    cnt = cnt + 1
                    
    def get_patch_gen(self, dataset, h, batch_size=16, exp="", shuffle=True, aug=True, w=None):
        if w is None: w = h
        from keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=360,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=True)

        if aug is False:
            train_datagen = ImageDataGenerator(
                rescale=1./255)

        dirpath = self.path("patch", dataset=dataset, cls="", cnt="", exp=exp)[:-4]
        print(dirpath)
        return train_datagen.flow_from_directory(dirpath,
            target_size=(h, w),
            batch_size=batch_size,
            shuffle=shuffle)

    def get_mimg(self, iid, dataset="test"):
        return AnnImg(img_path=self.path("mask", dataset=dataset, iid=iid))

    def get_score(self, dataset, type):
        ret = []
        for cl in range(5):
            filepath = self.path("ssd_score",dataset=dataset, type=type, cname=CLASS_NAMES[cl])
            df = pd.read_csv(filepath, delimiter=" ", header=-1)
            arr = df.as_matrix()
            ret.append(self.getbbox(cl,arr))
        ret = np.concatenate(ret,axis=0)
        return ret
        
    def getbbox(self, cl, arr):
        ret = []
        for fname, conf, xmin, ymin, xmax, ymax in arr:
            iid = int(fname.split("/")[-2])
            x,y = fname.split("/")[-1][:-4].split("_")
            x,y = int(x),int(y)
            xmin, ymin, xmax, ymax  = xmin+x, ymin+y, xmax+x, ymax+y
            ret.append([iid,cl,int(conf*100),xmin,ymin,xmax,ymax])
        return np.array(ret)
    
   
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
        
    def dataset(self, dataset, test=None, ann_chk=True):
        if self.type=="datasource": 
            return AnnImgsDir(self.path("dataset", dataset=dataset), test=test, ann_chk=ann_chk)
        else: 
            assert(self.type=="tiles_dir")
            return AnnImgsTilesDir(self.path("dataset", dataset=dataset))
    
    def dataset_path(self, dataset):
        return (self.path("dataset", dataset=dataset))
     
    def trainval(self, split=0.2, dataset="all"):
        all = self.dataset(dataset)
        train, val = all.split(split)
        val.savedir(self.path("dataset",dataset="val"))
        train.savedir(self.path("dataset",dataset="train"))
    
    
    def split_half(self, dataset="all", half=0):
        all = self.dataset(dataset)
        arr = all.counts(take=50)
        all = self.dataset(dataset)
        all.iids = arr[:,0]
        def prep(aimg):
            aimg = aimg.bound().fpups().bcrop()[half]
            aimg = aimg.bound()
            return aimg
        all.savedir(self.path("dataset",dataset=dataset+"_half"+str(half)),prep=prep)
    
    
    def scale(self, dataset="all"):
        all = self.dataset(dataset)
        def prep(aimg):
            return aimg.scale(100/aimg.ann.get_scale())
        all.savedir(self.path("dataset",dataset=dataset+"_scaled"),prep=prep)
    
    def plot_dataset(self, dataset):
        dirpath = self.path("plot_ds",dataset=dataset)
        os.makedirs(dirpath, exist_ok=True)
        ds = self.dataset(dataset)
        ds.save_plot(dirpath+os.sep+"{iid}.jpg")
    
    @property
    def all(self):
        return self.dataset("all")
    
    @property
    def train(self):
        return self.dataset("train")
    
    @property
    def trainpath(self):
        return self.dataset_path("train")
    
    @property
    def valpath(self):
        return self.dataset_path("val")
    
    @property
    def val(self):
        return self.dataset("val")
    
    @property
    def testpath(self):
        return self.dataset_path("test")
    
    @property
    def test(self):
        return self.dataset("test")

    def sub(self, dataset="test", inp="sample_submission", out="sub", start=0, end=18635, conf=95, th=0.33):
        inpp = self.path("sub",dataset=dataset,sub=inp)
        test = self.dataset("test", test=True)
        ret = np.loadtxt(inpp, delimiter=',', skiprows=1).astype('int32')
        for iid in tqdm(test.iids[start:end]):
            aimg,_ = test.aimg(iid)
            if aimg.ann is None: continue
            aimg = aimg.fconf(conf)
            if aimg.count == 0: continue
            aimg = aimg.allNMS(th)
            ret[iid,1:] = aimg.counts
        outp = self.path("sub",dataset=dataset,sub=out)
        np.savetxt(outp, ret, delimiter=',', comments='', fmt='%d',
           header='test_id,adult_males,subadult_males,adult_females,juveniles,pups')
