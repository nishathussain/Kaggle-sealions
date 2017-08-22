from PIL import Image, ImageDraw
import numpy as np
from polarbear.ann import Ann
#from polarbear import CLASS_NAMES, CLASS_COLORS
from random import seed, random, randint
from scipy.spatial import Voronoi
import torch
from torch.autograd import Variable
from itertools import chain, islice
from skimage.morphology import watershed
from scipy.spatial import Voronoi

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

class AnnImg(object):
    
    def __init__(self, img=None, ann=None, **kwargs):
        if not img is None:
            self.img = img
        elif "npimg" in kwargs:
            self.img = Image.fromarray(kwargs["npimg"].astype('uint8'))
        else: self.img = Image.open(kwargs["img_path"])

        if ann is not None: self.ann = ann
        elif "ann_path" in kwargs: 
            scale = kwargs.get("scale",None)
            self.sc = scale
            #print("AnnImg scale", scale)
            self.ann = Ann(file=kwargs["ann_path"], scale=scale)
        else: self.ann = None
        
    @classmethod        
    def open(cls, img_path, ann_path=None, scale=None):
        
        ann = None
        if not ann_path is None:
            ann = Ann(file=ann_path,scale=scale)
        return AnnImg(img, ann)
        
    def crop(self, x1, y1, x2, y2, truncate=False, overlap=True):
        img = self.img.crop((x1,y1,x2,y2))
        ann = None
        if self.ann is not None:
            ann = self.ann.crop(x1,y1,x2,y2, truncate=truncate, overlap=overlap)
        return AnnImg(img, ann)
    
    def cropH(self):
        W,H = self.WH
        img1 = self.crop(0,0,W,H//2)
        img2 = self.crop(0,H//2,W,H)
        return img1, img2
    
    def cropW(self):
        W,H = self.WH
        img1 = self.crop(0,0,W//2,H)
        img2 = self.crop(W//2,0,W,H)
        return img1, img2
    
    def cropHW(self):
        W,H = self.WH
        img1, img2 = self.cropW()
        img11, img12 = img1.cropH()
        img21, img22 = img2.cropH()
        return img11, img12, img21, img22
    
    def fpups(self):
        return AnnImg(self.img, self.ann.fpups())

    def fconf(self, p):
        return AnnImg(self.img, self.ann.fconf(p))

    def kconf(self, p):
        return AnnImg(self.img, self.ann.kconf(p))

    def allNMS(self, th):
        return AnnImg(self.img, self.ann.allNMS(th))

    @property
    def counts(self):
        return self.ann.counts

    @property
    def count(self):
        return self.ann.count

    def oneclass(self):
        return AnnImg(self.img, self.ann.oneclass())
    
    def setbox(self, size):
        return AnnImg(self.img, self.ann.setbox(size))
    @property
    def cxy(self):
        return self.ann[:,0], self.ann[:,1], self.ann[:,2]
    
    def hm_pn(self, p=50, bg=None):
        aimg = self
        fp = aimg.hneg(th=True).fconf(p).ann
        #total = aimg.fcenter().ann
        total = aimg.fcenter().hmconf().ann
        tp = total.fconf(p)
        fn = total.kconf(p)
        fn.cl = 4
        plot = None
        ann = tp.append(fn).append(fp)
        if bg is not None: plot = bg.wann(ann).plot(rect=False).img
        else: plot = aimg.wann(ann).plot(rect=False).img
        return ann, plot
    
    def vor(self, mind=None, maxd=None):
        return self.wann(self.ann.vor(mind,maxd))    
    
    def overlay(self, aimg):
        bg = self
        np1 = bg.np(t=False).copy()
        np2 = aimg.np().transpose((1,2,0))
        c = 1
        print(np1.shape, np2.shape)
        #np1 = np[:,:,c]
        np1[np2[:,:,0]<100] = (np1[np2[:,:,0]<100] * 0.5 + (np2[np2[:,:,0]<100]) * 0.5)
        return AnnImg(npimg=np1)
    
    @property
    def WH(self):
        return self.img.size
    
    @property
    def count(self):
        return self.ann.count
    
    def cropd(self, x, y, d):
        return self.crop( x, y, x+d, y+d)
    
    def cropdd(self, x, y, d):
        return self.crop( x-d, y-d, x+d, y+d)
    
    def rcropdd(self, d):
        W,H = self.WH
        return self.cropdd(randint(0,W), randint(0,H), d)
    
    def bcropW(aimg, d=100):
        W,H = aimg.WH
        xsort = np.sort(aimg.ann.xy[:,0])
        w = xsort[xsort.shape[0]//2]
        return aimg.crop(0,0,w+d,H),aimg.crop(w-d,0,W,H),(w/W)

    def bcropH(aimg, d=100):
        W,H = aimg.WH
        ysort = np.sort(aimg.ann.xy[:,1])
        h = ysort[ysort.shape[0]//2]
        return aimg.crop(0,0,W,h+d), aimg.crop(0,h-d,W,H), (h/H)    
    
    def bcrop(aimg, d=100):
        w1,w2,ww = aimg.bcropW(d)
        h1,h2,hh = aimg.bcropH(d)
        if(abs(0.5-ww)<abs(0.5-hh)): return w1,w2
        else: return h1,h2
    
    def collate_cls(self, cl, limit=8):
        ret = []
        for c,ai in self.crop_ann():
            if c==cl:
                ret.append(ai.np(t=False))
            if len(ret)==8: break
        if len(ret)==0: return None
        return AnnImg(npimg=np.concatenate(ret, axis=1)).img
    
    def collate(self, limit=8):
        ret = []
        for i in range(5):
            a = self.collate_cls(i, limit)
            if a is not None: ret.append((i,a))
        return ret
    def nimg(self, d, ratio=1):
        numneg = int(self.count * ratio)
        W,H = self.WH
        
        arr = []
        steps = 0
        while len(arr) < numneg and steps<10000:
            steps += 1
            x,y = (randint(0,W), randint(0,H))
            if self.cropdd(x,y,d).count == 0:
                  arr.append((-1,x,y))
        ann = Ann(dets=np.array(arr))
        return AnnImg(self.img, ann)
    
    def crop_ann(self, d=None, cls=None):
        for cl,conf,xmin,ymin,xmax,ymax in self.ann.gen_dets():
            if cls is not None and cls != cl: continue
            if d is not None:
                yield cl, self.cropdd((xmin+xmax)//2,(ymin+ymax)//2, d)
            else:
               yield cl, self.crop(xmin,ymin,xmax,ymax)
    
    def siamese(self, d, cl1, cl2, cnt=2):
        for clx,x in list(self.crop_ann(d, cl1)):
            self.ann.shuffle()
            for cly,y in islice(self.crop_ann(d, cl2),cnt):
                yield x,y,clx==cly

    def siamese23(self, d):
        return chain(self.siamese(d,2,2),self.siamese(d,2,3),self.siamese(d,3,2),self.siamese(d,3,3))
    
    def blank(self, r=5, ratio=1):
        img = Image.new('RGBA', self.img.size, (0,0,0,255))
        if r is None: return AnnImg(img, self.ann).plot(rect=False, label=False, color=(255,255,255), bound=True, ratio=ratio)
        else: return AnnImg(img, self.ann).plotc(color=(255,255,255,255), r=r)

    def clear(self, clr = (0,0,0,255)):
        return AnnImg(Image.new('RGBA', self.img.size, (0,0,0,255)), self.ann)

    def crop_neg(self, d, num):
        while num>0:
            aimg = self.rcropdd(d)
            if aimg.count == 0:
                num -= 1
                yield aimg
    
    def neg_mask(self, ratio=1):
        return AnnImg(npimg=self.blank(None, ratio=ratio).np()[0])
    
    def mergeW(x, y):
        return AnnImg(npimg=np.concatenate([x.np(t=False),y.np(t=False)],axis=1))

    def crop_posneg(self, d=None):
        cnt = 0
        Wmax = 0
        for cl, aimg in self.crop_ann(d):
            cnt += 1
            W,H = aimg.WH
            Wmax = max(Wmax,W)
            yield cl, aimg
        for aimg in self.crop_neg(Wmax//2, max(cnt,50)):
            yield -1, aimg

    def cropc(self, d):
        x,y = self.WH
        x,y = x//2, y//2
        return self.crop( x-d, y-d, x+d, y+d)
    
    def bound(self):
        xmin,ymin,xmax,ymax = self.ann.bound()
        return self.crop(xmin, ymin, xmax, ymax)
    
    def posmid(self, d, dist=None):
        if dist is None: dist = 1.5*d
        pos = list([aimg.np() for cl,aimg in self.crop_ann(d)])
        dets,_ = self.ann.midd(dist)
        ann = Ann(dets=dets)
        print("posmid", len(pos), ann.count)
        aimg = AnnImg(self.img, ann)
        mid = list([aimg.np() for cl,aimg in aimg.crop_ann(d)])
        Xp, Xn = np.stack(pos), np.stack(mid)
        Yp, Yn = np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])
        return np.concatenate([Xp,Xn]), np.concatenate([Yp,Yn]), Xp, Xn
    
    def dbox(self, d):
        """
        Increase the size of boxes by 2*d
        """
        return self.wann(self.ann.dbox(d))
        
    def posneg(self, d):
        pos = list([aimg.np() for cl,aimg in self.crop_ann(d)])
        cl = list([cl for cl,aimg in self.crop_ann(d)])
        #for i in range(len(pos)):
        neg = []
        seed(0)
        while len(neg)<len(pos):
            aimg = self.rcropdd(d)
            if aimg.ann.count==0: 
                neg.append(aimg.np())
        Xp, Xn = np.stack(pos), np.stack(neg)
        Yp, Yn = np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])
        Yp, Yn = np.array(cl)+1, np.zeros(Xn.shape[0])
        return np.concatenate([Xp,Xn]), np.concatenate([Yp,Yn]), Xp, Xn
    
    def posnegHW(self, d):
        i1,i2,i3,i4 = self.cropHW()
        X1, Y1, _, _ = i1.posneg(d)
        X2, Y2, _, _ = i2.posneg(d)
        X3, Y3, _, _ = i3.posneg(d)
        X4, Y4, _, _ = i4.posneg(d)
        Xt = np.concatenate([X1,X3])
        Yt = np.concatenate([Y1,Y3])
        Xv = np.concatenate([X2,X4])
        Yv = np.concatenate([Y2,Y4])
        return Xt,Yt,Xv,Yv
    
    def posmidHW(self, d):
        i1,i2,i3,i4 = self.cropHW()
        X1, Y1, _, _ = i1.posmid(d)
        X2, Y2, _, _ = i2.posmid(d)
        X3, Y3, _, _ = i3.posmid(d)
        X4, Y4, _, _ = i4.posmid(d)
        Xt = np.concatenate([X1,X3])
        Yt = np.concatenate([Y1,Y3])
        Xv = np.concatenate([X2,X4])
        Yv = np.concatenate([Y2,Y4])
        return Xt,Yt,Xv,Yv
    
    
    def tile(self, d, s):
        W,H = self.WH
        for x in range(0,W,s):
            for y in range(0,H,s):
                yield self.cropd(x,y,d),x,y

    def mtile(aimg, mimg, d, s, th=0.9):
        W,H = aimg.WH
        if mimg is None: 
            mimg = AnnImg(npimg=np.zeros((W,H)))
        mimg = mimg.resize(W,H)
        for a,x,y in aimg.tile(d, s):
            m = mimg.cropd(x,y,d)
            p = 1-(m.np().min()/255.0)
            if(p>th):
                yield a,x,y
            #print("skip",x,y,p)

    def merge(self, fun, d=900, s=800, mimg=None):
        ret = []
        for timg,x,y in self.mtile(mimg, d,s):
            ann = fun(timg)
            if ann is None: continue
            ann.dxy(-x,-y)
            ret.append(ann.dets)
        ret = np.concatenate(ret)   
        return Ann(dets=ret)

    def test(self, model, d=900, s=800, cuda=True, mimg=None):
        model.eval()
        model.phase = "test"
        def fun(timg):
            img, tgt = timg.ssd()
            if cuda: img = img.cuda()
            dets = model.forward(Variable(img))
            dim = img.size(2)
            dets = dets[0].data
            if cuda: dets = dets.cpu()
            ann = Ann(tensor=dets, dim=img.size(2))
            return ann
        return self.merge(fun,d,s,mimg=mimg)

    def resize(self, x, y=None):
        if y is None: y = x
        W,H = self.img.size
        xs,ys = x/W,y/W
        ann = None
        if self.ann is not None:
            ann = self.ann.resize(x,y, W, H)
        return AnnImg(self.img.resize((x,y)), ann)
    
    def scale(self, s):
        x,y = self.WH
        return self.resize(int(x*s), int(y*s))
    
    def pad(self, padx, pady=None):
        if pady is None: pady=padx
        W,H = self.WH
        return self.crop(-padx,-pady,W+padx,H+pady)
    
    def plt(self):
        img = self.img.copy()
        draw = ImageDraw.Draw(img)
        for c,x,y in zip(*self.cxy):
            draw.ellipse((x-d,y-d,x+d,y+d), fill=CLASS_COLORS[c])
        return AnnImg(img, self.ann)
    
    def plot(self, ann=None, rect=True, label=True, color=None, save=None, r=5, bound=False, ratio=1):
        if self.ann is None: return AnnImg(self.img)
        if ann is not None:
            aimg = AnnImg(self.ann.plot(self.img, rect=False, text=label, color=color, r=r, bound=bound, ratio=ratio), ann)
            return aimg.plot()
        else: 
            aimg = AnnImg(self.ann.plot(self.img, rect=rect, text=label, color=color, r=r, bound=bound, ratio=ratio), self.ann)
            if save is not None: aimg.img.save(save)
            return aimg
    
    def plotc(self, color=None, save=None, r=5, bound=False):
        if self.ann is None: return self.img
        aimg = self.plot(rect=False, label=False, color=color, r=r, bound=bound)
        return aimg.img

    def unet_mask(self, minr=0.5, maxr=1.5, vor=None):
        cimg = self
        img1 = cimg.neg_mask(ratio=minr)
        img2 = cimg.neg_mask(ratio=maxr)
        arr = (img1.np() + img2.np())//2
        aimg = AnnImg(Image.fromarray(arr[0].astype('uint8'))).wann(self.ann)
        if vor is not None:
            
            aimg2 = AnnImg(aimg.plot_vor(width=vor))
            if aimg2.np().sum() == 0: 
                aimg2 = AnnImg(aimg.plot_vor(width=1))
            aimg = aimg2         
        return aimg

    def _repr_png_(self):
        return self.plot().img._repr_png_()

    def save(self, img_path, ann_path=None):
        if ann_path is not None and self.ann is not None:
            self.ann.save(ann_path)
        self.img.save(img_path)
        
    def np(self, t=True):
        im = self.img
        arr = np.asarray(im, dtype = np.float)
        if len(arr.shape)==3:
            if t: return np.transpose(arr, (2,0,1))
            else: return arr
        else: return np.expand_dims(arr, 0)

    def negative(self, p=0.2):
        arr = self.np()
        arr = (arr<(p*255)).astype('int32')
        img = Image.fromarray(np.uint8(arr[0]*255))
        return AnnImg( img, self.ann.copy())

    def setScale(self, scale=50):
        return AnnImg(self.img, self.ann.setScale(scale))
    
    
    def rescale(self, scale=50):
        ratio = scale/self.sc
        return self.scale(ratio)
    
    def ssd(aimg):
        tile, _ = aimg.WH
        target = None
        if aimg.ann is not None:
            target = aimg.ann.ssd(tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()
        return img.unsqueeze(0), target
    
    def hist(self):
        img = self.np()
        ret = []
        for i in range(img.shape[0]):
            hist,bins = np.histogram(img[i].flatten(),256,[0,256])
            ret.append(hist)
        return np.concatenate(ret)
        
    def hist2(self):
        ret = []
        W,H = self.WH
        for img in self.cropHW():
            ret.append(img.hist())
        return np.concatenate(ret)
        
    
    def dist(self, min=None, max=None):
        W,H = self.WH
        arr = self.ann.dist(W,H,min,max)
        img = Image.fromarray(arr.astype('uint8'))
        return AnnImg(img, self.ann)
    
    def neg(self, ratio=1.0, num=None, minn=None):
        if num is None: num = int(self.count * ratio)
        if minn is not None: num = max(num, minn) 
        mask = self.neg_mask().inv().np()
        #print("mask", mask.shape)
        y,x = mask[0].nonzero()
        pts = np.stack([x,y],axis=1)
        np.random.shuffle(pts)
        xy = pts[:num]
        cl = -1*np.ones((xy.shape[0],1))
        ann = Ann(dets=np.concatenate([cl,xy],axis=1).astype('int32'))
        return self.wann(ann)
    
    def hneg(self, ratio=1.0, num=None, minn=None, th=False, conf=10):
        if num is None: num = int(self.count * ratio)
        if minn is not None: num = max(num, minn) 
        mask = self.neg_mask().inv().np()
        #print("mask", mask.shape)
        y,x = mask[0].nonzero()
        hm = self.np()[0]
        #print("hm", hm.shape)
        idx = np.argsort(hm[y,x])
        py,px = y[idx],x[idx]
        xy = np.stack([px,py],axis=1)
        cl = -1 * np.ones((xy.shape[0],1))
        h = hm[xy[:,1],xy[:,0]]
        
        ann = Ann(dets=np.concatenate([cl.reshape(-1,1),xy],axis=1).astype('int32'))
        ann.conf = ((255-h)*100)/255
        ann = ann.fconf(conf)
        #print(ann)
        if ann.count < num: return self.wann(ann)
        if th: ann = ann.pnms(50)
        return self.wann(ann.take(num))
    
    def hmconf(self):
        return self.wann(self.ann.hmconf(self.np()[0]))
    
    def wann(self, ann):
        return AnnImg(self.img, ann)
    
    def append(self, ann):
        return AnnImg(self.img, self.ann.append(ann))
    
    def inv(self):
        return AnnImg(npimg=(255-self.np()[0]).astype('uint8'),ann=self.ann)
    
    def threshold(self, th):
            return AnnImg(npimg=(self.np()[0]<th)*255)
    
    def watershed(self, th=128):
        distance1 = self.np()[0]
        W,H = self.WH
        markers1 = np.zeros((W,H))
        for i,(x,y) in enumerate(self.ann.xy):
            markers1[y][x] = i
        image1 = distance1<th
        labels1 = watershed(distance1, markers1, mask=image1)
        return labels1
    
    def fcenter(self):
        W,H = self.WH
        return self.wann(self.ann.fcenter(0,0,W,H))
    
    def norm(self):
        npimg = self.np()
        mn = npimg.min()
        mx = npimg.max()
        out = 255*(npimg-mn)/(mx-mn)
        return AnnImg(npimg=out[0], ann=self.ann)
    
    def cuts(self, labels1):
        XX = labels1.copy()
        X = labels1.copy()
        XX.fill(0)
        for i in range(1,X.shape[0]-1):
            for j in range(1,X.shape[1]-1):
                if X[i][j] != 0:
                    if X[i][j] != X[i+1][j] and X[i+1][j]!=0:
                        XX[i][j] = 1
                    if X[i][j] != X[i-1][j] and X[i-1][j]!=0:
                        XX[i][j] = 1
                    if X[i][j] != X[i][j+1] and X[i][j+1]!=0:
                        XX[i][j] = 1
                    if X[i][j] != X[i][j-1] and X[i][j-1]!=0:
                        XX[i][j] = 1
        return Image.fromarray(XX.astype('uint8')*255)
    
    def plot_vor(cimg, color=0, width=4):
        #cimg = cimg.copy()
        points = cimg.ann.xy
        vor = Voronoi(points)
        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)

        finite_segments = []

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max()

                finite_segments.append([vor.vertices[i], far_point])
                
        #print(finite_segments)

        im=cimg.img.copy()
        draw = ImageDraw.Draw(im)
        for p1,p2 in finite_segments:
            draw.line((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])), fill=color, width=width)
        return im