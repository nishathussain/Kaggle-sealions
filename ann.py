#from polarbear import CLASS_NAMES, CLASS_COLORS
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
RATIOS=np.array([1.00,0.80,0.70,0.60,0.40,0.50])
import os
import torch
from scipy.spatial import Voronoi
import cv2
from layers.box_utils import nms, match, jaccard
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform 
from sklearn.metrics.pairwise import pairwise_distances
from itertools import product , chain

CLASS_NAMES = (
            'am',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups',
            'bg')


CLASS_NUM = {
            'adult_males' : 0,
            'subadult_males': 1,
            'adult_females': 2,
            'juveniles': 3,
            'pups': 4, 
            'bg': -1}
        
CLASS_COLORS = (
            (255,0,0, 128),          # red
            (250,10,250,128),       # magenta
            (84,42,0,128),          # brown 
            (30,60,180,128),        # blue
            (35,180,20,128),        # green
            (0,0,0,128),            # background
            )

NUM_CLASSES = 5

from sklearn.metrics import precision_recall_curve

class Ann(object):
    
    #cl, conf, xmin, ymin, xmax, ymax
    def __init__(self, **kwargs):
        if "dets" in kwargs: 
            
            dets = kwargs["dets"]
            
            if dets is None: dets=np.zeros((0,6))
            if(len(dets.shape)==1): 
                if(dets.shape[0]==0): dets = np.zeros((0,6))
                else: dets = np.expand_dims(dets,axis=0)
            assert(len(dets.shape)==2)
            assert(dets.shape[1]==3 or dets.shape[1]==6 or dets.shape[1]==4 or dets.shape[1]==5)
            if dets.shape[1]==3: 
                scale = 50
                if "scale" in kwargs: 
                    if kwargs["scale"] is not None:
                        scale = int(kwargs["scale"])
                        self.scale = scale
                #print("setting scale", scale)
                self.sc = scale
                self.dets = self.set_cxy(dets,scale)

            if dets.shape[1]==5: 
                N = dets.shape[0]
                c = dets[:,-1].copy()
                dets *= kwargs["dim"]
                #print(c)
                conf = np.ones((N,))*100
                self.dets = np.stack([c,conf,dets[:,0],dets[:,1],dets[:,2],dets[:,3]]).transpose().astype('int32')

            if dets.shape[1]==4: 
                N = dets.shape[0]
                c = np.zeros((N,))
                conf = np.ones((N,))*100
                self.dets = np.stack([c,conf,dets[:,0],dets[:,1],dets[:,2],dets[:,3]]).transpose().astype('int32')
                #print(self.dets)
            elif dets.shape[1]==6: self.dets = dets.astype('int32')
        elif "tensor" in kwargs: Ann.torch_init(self, kwargs["tensor"], kwargs["dim"])
        elif "xml" in kwargs: Ann.xml(self, kwargs["xml"])
        elif "file" in kwargs: 
            #print("file", kwargs["file"], kwargs.get("scale",None))
            Ann.file(self, kwargs["file"], scale=kwargs.get("scale",None))
    
        
    def set_cxy(cls, cxy, scale=50):
        assert( len(cxy.shape)==2 and cxy.shape[1]==3)
        if scale==0: scale=50
        N = cxy.shape[0]        
        cl = cxy[:,0]
        x = cxy[:,1]
        y = cxy[:,2]
        d = scale * RATIOS
        scales = d[cl]
        dets = [cl, (cl*0)+100, x-scales, y-scales, x+scales, y+scales]
        dets = np.stack(dets,axis=1).astype('int32')
        return dets
        
    def get_scale(self):
        dets0 = self.dets[self.dets[:,0]==0]
        if dets0.shape[0]==0: return 100
        else: return dets0[0,4] - dets0[0,2]
        
    @property
    def count(self):
        return self.dets.shape[0]
    
    def xml(self, xml_path):
        dets = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root.findall('object'):
            label = child.find('name').text
            for box in child.findall('bndbox'):
                dets.append([ CLASS_NUM[label], 100, 
                        int(box.find('xmin').text), 
                        int(box.find('ymin').text), 
                        int(box.find('xmax').text), 
                        int(box.find('ymax').text)])
        #print(dets)
        Ann.__init__(self, dets=np.array(dets))

    def torch_init(self, tensor, dim=300):
        if tensor.dim() == 2: Ann.__init__(self, dets=tensor.numpy())
        else: 
            assert(tensor.dim()==3)
            ret = []
            for i in range(1,tensor.size(0)):
                cdets = tensor[i]
                #print(cdets)
                idx = torch.nonzero(cdets[:,0]>0.5)
                if idx.nelement()==0:continue
                #print(idx)
                d = cdets.index_select(0,idx.squeeze())
                cl = torch.ones((idx.nelement(),1))*(i-1)
                #print("d", d.type(), cl.type())
                d = torch.cat([cl.cpu() , d[:,0:1]*100, d[:,1:]*dim], 1).round()
                ret.append(d)
            if len(ret)==0: ret = torch.zeros((0,6)).cpu()
            else: ret = torch.cat(ret,0)
            Ann.__init__(self, dets=ret.cpu().numpy())

    @property    
    def pts(self):
        for cl,conf,xmin,ymin,xmax,ymax in self.gen_dets():
            yield round((xmin+xmax)/2), round((ymin+ymax)/2)
            
    
    def append(self, that):
        return Ann(dets=np.concatenate([self.dets,that.dets]))
    
    @property    
    def xy(self):
            x = np.round((self.xmin+self.xmax)/2) 
            y = np.round((self.ymin+self.ymax)/2)
            return np.stack([x,y],axis=1).astype('int32')
        
    @property
    def mid(self):
        vor = Voronoi(list(self.pts))
        tmp = []
        for i,j in vor.ridge_points:
            pi = vor.points[i]
            pj = vor.points[j]
            p = (pi+pj)//2
            tmp.append((0,p[0],p[1]))
        tmp = np.array(tmp)
        return tmp
    
    def take(self, num):
        return Ann(dets=self.dets[:num])
    
    def midd(self, dist=100):
        pts = []
        ann1 = self
        ret = []
        for (xi,yi) in ann1.pts:
            for (xj,yj) in ann1.pts:
                d =( (xi-xj)**2 + (yi-yj)**2 ) ** 0.5
                if not ((xi==xj) and (yi==yj)):
                    ret.append([d,(xi+xj)//2,(yi+yj)//2])
                    if d<dist: pts.append([0,(xi+xj)//2,(yi+yj)//2])
                        
        ret = np.array(ret)
        #np.sort()
        return np.array(pts), ret
        
    def fpups(self):
        dets = self.dets
        return Ann(dets=dets[dets[:,0]!=4])
        
    def fclass(self, cl):
        dets = self.dets
        return Ann(dets=dets[dets[:,0]==cl])
               
    def oneclass(self, cls=0):
        self = self.copy()
        self.cl *= 0
        return Ann(dets=self.dets)
    def setbox(self, size):
        self = self.copy()
        x = (self.xmin+self.xmax)//2
        y = (self.ymin+self.ymax)//2
        self.xmin = x - (size)//2
        self.xmax = x + (size)//2
        self.ymin = y - (size)//2
        self.ymax = y + (size)//2
        return self
 
    def file(cls, file_path, scale=None):
        ext = os.path.splitext(file_path)[-1]
        #print(ext)
        if ext==".xml": 
            Ann.xml(cls, file_path)
        else: 
            assert(ext==".csv") 
            ann = np.loadtxt(file_path, delimiter=",").astype('int32')
            if(len(ann.shape)==1): ann = np.expand_dims(ann,axis=0)
            assert(len(ann.shape)==2 and (ann.shape[1]==3 or ann.shape[1]==6))
            #print("dets", ann.shape, scale)
            Ann.__init__(cls, dets=ann, scale=scale)
    
    @property
    def cl(self): return self.dets[:,0]
    @cl.setter
    def cl(self, value): self.dets[:,0] = value
    @property
    def conf(self): return self.dets[:,1]
    @conf.setter
    def conf(self, value): self.dets[:,1] = value.astype('int32')
    @property
    def xmin(self): return self.dets[:,2]
    @xmin.setter
    def xmin(self, value): self.dets[:,2] = value
    @property
    def xmax(self): return self.dets[:,4]
    @xmax.setter
    def xmax(self, value): self.dets[:,4] = value
    @property
    def ymin(self): return self.dets[:,3]
    @ymin.setter
    def ymin(self, value): self.dets[:,3] = value
    @property
    def ymax(self): return self.dets[:,5]
    @ymax.setter
    def ymax(self, value): self.dets[:,5] = value
    
    def getCounts(self):
        return Counter(self.cl)
        
    def gen_dets(self):
        for i in range(self.dets.shape[0]):
            yield self.dets[i,0],self.dets[i,1],self.dets[i,2],self.dets[i,3],self.dets[i,4],self.dets[i,5]
            
    def get_cols(self):
        return self.dets[:,0],self.dets[:,1],self.dets[:,2],self.dets[:,3],self.dets[:,4],self.dets[:,5]
    
    def get_dxy(self, dx, dy):
        cl,conf,xmin,ymin,xmax,ymax = self.get_cols()
        return Ann().init_cols(cl, conf, xmin+x, ymin+y, xmax+x, ymax+y)
    
    def crop(self, XMIN, YMIN, XMAX, YMAX, truncate=False, overlap=False):
        cl, conf, xmin, ymin, xmax, ymax = self.get_cols()
        x = (xmin+xmax)//2
        y = (ymin+ymax)//2
        mask = (x>=XMIN) & (y>=YMIN) & (x<=XMAX) & (y<=YMAX)
        if overlap:
            mask = ((xmax<=XMIN) | (xmin>=XMAX) | (ymax<=YMIN) | (ymin>=YMAX))==False
            
        #mask1 = (xmin>=XMIN) & (ymin>=YMIN) & (xmin<=XMAX) & (ymin<=YMAX)
        #mask2 = (xmax>=XMIN) & (ymax>=YMIN) & (xmax<=XMAX) & (ymax<=YMAX)
        dets = self.dets[mask] #1 | mask2]
        ann = Ann(dets=dets)
        if truncate:
            ann.xmin = np.minimum(np.maximum(ann.xmin, XMIN), XMAX)
            ann.ymin = np.minimum(np.maximum(ann.ymin, YMIN), YMAX)
            ann.xmax = np.minimum(np.maximum(ann.xmax, XMIN), XMAX)
            ann.ymax = np.minimum(np.maximum(ann.ymax, YMIN), YMAX)
        ann.dxy(XMIN, YMIN)
    
        return ann
    
    def fcenter(self, XMIN, YMIN, XMAX, YMAX):
        cl, conf, xmin, ymin, xmax, ymax = self.get_cols()
        x = (xmin+xmax)//2
        y = (ymin+ymax)//2
        mask = (x>=XMIN) & (y>=YMIN) & (x<XMAX) & (y<YMAX)
        return Ann(dets=self.dets[mask])
    
    def crop2(self, XMIN, YMIN, XMAX, YMAX, truncate=False):
        cl, conf, xmin, ymin, xmax, ymax = self.get_cols()
        mask = (xmax<=XMIN) | (xmin>=XMAX) | (ymax<=YMIN) | (ymin>=YMAX)
        #mask1 = (xmin>=XMIN) & (ymin>=YMIN) & (xmin<=XMAX) & (ymin<=YMAX)
        #mask2 = (xmax>=XMIN) & (ymax>=YMIN) & (xmax<=XMAX) & (ymax<=YMAX)
        dets = self.dets[mask==False] #1 | mask2]
        ann = Ann(dets=dets)
        if truncate:
            ann.xmin = np.minimum(np.maximum(ann.xmin, XMIN), XMAX)
            ann.ymin = np.minimum(np.maximum(ann.ymin, YMIN), YMAX)
            ann.xmax = np.minimum(np.maximum(ann.xmax, XMIN), XMAX)
            ann.ymax = np.minimum(np.maximum(ann.ymax, YMIN), YMAX)
        ann.dxy(XMIN, YMIN)
        return ann
    
    
    def resize(self, x, y, X, Y=None):
        if Y is None: Y = X
        ann = self.copy()
        ann.xmin = np.round((ann.xmin*x)/X)
        ann.xmax = np.round((ann.xmax*x)/X)
        ann.ymin = np.round((ann.ymin*y)/Y)
        ann.ymax = np.round((ann.ymax*y)/Y)
        return ann
        
    def dxy(self, x, y):
        self.xmin -= x
        self.xmax -= x
        self.ymin -= y
        self.ymax -= y
        return self
    
    
    def dbox(self, d):
        """
        Increase the size of boxes by 2d 
        """
        self = self.copy()
        self.xmin += d
        self.xmax -= d
        self.ymin += d
        self.ymax -= d
        return self

    def pad(self, padx, pady=None):
        if pady is None: pady = padx
        ann = self.copy()
        ann.dxy(-padx, -pady)
        return ann
        
    def __str__(self):
        return self.dets.__str__()
        
    def __repr__(self):
        return self.dets.__repr__()
        
    def cropd(self, x, y, d, truncate=True):
        return self.crop( x, y, x+d, y+d, truncate)
    
    def cropdd(self, x, y, d, truncate=True):
        return self.crop( x-d, y-d, x+d, y+d, truncate)
    
    #crop center of image of width = d+d, height = d+d
    def cropc(self, d, truncate=True):
        x,y = self.WH
        x,y = x//2, y//2
        return self.crop( x-d, y-d, x+d, y+d, truncate)
    
    def plot(self, image, rect=True, text=True, color=None, r=5, bound=False, ratio=1):
        img = image.copy()
        if img.getbands()==('L',) and color is None: color = (0)
        draw = ImageDraw.Draw(img)
        for c,conf,xmin,ymin,xmax,ymax in self.gen_dets():
            if color is None: clr = CLASS_COLORS[c]
            else:  clr = color
            if rect: 
                #print("xy", xmin, ymin, xmax, ymax)
                draw.rectangle((xmin,ymin,xmax,ymax), outline=clr)
                if text: draw.text((xmin,ymin-10),CLASS_NAMES[c]+":"+str(conf),  fill=clr)
            else: 
                x = (xmin+xmax)//2
                y = (ymin+ymax)//2
                d = r
                if bound: d = ((xmax-xmin)//2) * ratio
                draw.ellipse((x-d,y-d,x+d,y+d), fill=clr)
                if text: draw.text((x,y-15),CLASS_NAMES[c]+":"+str(conf),  fill=clr)
        return img
    
    def unique(self):
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        wh = set(sorted(list(zip(w.tolist(), h.tolist()))))
        return wh

    def plot_size(self, image, rect=True, text=False, color=None):
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        wh = set(sorted(list(zip(w.tolist(), h.tolist()))))
        #print("wh", wh)
        ret = []
        for ww,hh in wh:
            mask = (w==ww) & (h==hh)
            dets = (self.dets[mask])
            ret.append(Ann(dets=dets).plot(image,rect=rect,text=text,color=color))
        return ret
    

    def save(self, file_path):
        ext = os.path.splitext(file_path)[-1]
        #print(ext)
        if ext==".csv": self.save_csv(file_path)
        elif ext==".xml": self.save_xml(file_path)
        
    def save_csv(self, file_path):
        np.savetxt(file_path, self.dets, delimiter=",", fmt="%d")
    
    def get_xml(self, h=300, w=300, c=3):
        root = ET.Element("annotation")
        size  = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)
        
        for cl,conf,xmin,ymin,xmax,ymax in self.gen_dets():
            obj_elm  = ET.SubElement(root, "object")
            ET.SubElement(obj_elm, "name").text = CLASS_NAMES[cl]
            ET.SubElement(obj_elm, "pose").text= "0"
            ET.SubElement(obj_elm, "truncated").text = "0"
            ET.SubElement(obj_elm, "difficult").text = "0"

            bndbox = ET.SubElement(obj_elm, "bndbox")
            ET.SubElement(bndbox, "ymin").text = str(max(ymin,0))
            ET.SubElement(bndbox, "ymax").text = str(min(ymax,299))
            ET.SubElement(bndbox, "xmin").text = str(max(xmin,0))
            ET.SubElement(bndbox, "xmax").text = str(min(xmax,299))
        tree = ET.ElementTree(root)
        return tree
    
    def copy(self):
        return Ann(dets=self.dets.copy())
    
    def save_xml(self, file_path):
        tree = self.get_xml()
        tree.write(file_path)
     
    def ssd(self, dim=300):
        dets = self.dets
        boxes = dets[:,2:]/dim
        classes = dets[:,0:1]
        return torch.from_numpy(np.concatenate([boxes, classes], axis=1))

    def totorch(self):
        return torch.from_numpy(self.dets)

    def setScale(self, scale=50):
        self = self.copy()
        x = (self.xmin + self.xmax)//2
        y = (self.ymin + self.ymax)//2
        self.xmin = x - scale
        self.ymin = y - scale
        self.xmax = x + scale
        self.ymax = y + scale
        return self

    # Do pytorch NMS and match boxes
    def classNMS(self, th=0.1):
        ann1 = self.totorch()
        ret = []
        for i in range(self.dets[:,0].max()):
            mask = torch.nonzero(ann1[:,0]==i)
            if(mask.nelement()>0):
                clann = ann1.index_select(0, mask[:,0])
                idx, cnt = nms(clann[:,2:].float(), clann[:,1].float(), th, 20000)
                tmp = clann.index_select(0, idx[:cnt].cpu())
                if tmp.nelement()>0: ret.append(tmp)
        return Ann(dets=torch.cat(ret).numpy())
    
    def allNMS(self, th=0.1):
        ann1 = self.totorch()
        ##print(ann1)
        ret = []
        mask = torch.nonzero(ann1[:,0]!=-1)
        if(mask.nelement()>0):
            clann = ann1.index_select(0, mask[:,0])
            idx, cnt = nms(clann[:,2:].float(), clann[:,1].float(), th, 20000)
            tmp = clann.index_select(0, idx[:cnt].cpu())
            if tmp.nelement()>0: ret.append(tmp)
        return Ann(dets=torch.cat(ret).numpy())
    def shuffle(self):
        np.random.shuffle(self.dets)

    def prec_recall(self, that, th=0.5):
        none = torch.zeros((0,6))
        bgt = torch.from_numpy(self.dets).float()
        bpd = torch.from_numpy(that.dets).float()
        
        overlaps = jaccard(bgt[:,2:], bpd[:,2:])
        overlaps[overlaps<th] = 0
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        best_prior_idx = best_prior_idx.squeeze()
    
        tp_gt = torch.nonzero(best_prior_idx)
        if tp_gt.nelement() == 0: return 0, 0, (none, none, bgt, bpd)
        tp_gt_boxes = bgt.index_select(0,tp_gt.squeeze())
    
        tp_pd = torch.index_select(best_prior_idx, 0, tp_gt.squeeze())
        tp_pd_boxes = bpd.index_select(0,tp_pd.squeeze())
    
        fn = torch.nonzero(best_prior_idx==0)
        if fn.nelement() == 0: fn_boxes = none
        else :fn_boxes = bgt.index_select(0,fn.squeeze())
    
        fp = torch.ones((len(bpd))).cpu()
        fp = fp.index_fill_(0,tp_pd,0).nonzero()
        if fp.nelement() == 0: fp_boxes = none
        else: fp_boxes = bpd.index_select(0,fp.squeeze())
        prec = len(tp_pd)/(len(tp_pd)+len(fp))
        recall = len(tp_gt)/(len(tp_gt)+len(fn))
    
        return prec, recall, (tp_gt_boxes, tp_pd_boxes, fn_boxes, fp_boxes)

    def fconf(self, p):
        mask = self.conf>=p
        return Ann(dets=self.dets[mask])
    
    def kconf(self, p):
        mask = self.conf<=p
        return Ann(dets=self.dets[mask])

    @property
    def counts(self):
        classes = self.cl
        ret = np.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            ret[i] = (classes==i).sum()
        return ret.astype('int32')
    
    
    def pn(self):
        return self.fclass(0).count,self.fclass(-1).count,self.fclass(4).count
    
    def bound(self):
        xmin,ymin = self.xmin.min(),self.ymin.min()
        xmax,ymax = self.xmax.max(),self.ymax.max()
        return xmin,ymin,xmax,ymax


    def max_th(self):
        boxes = torch.from_numpy(self.dets[:,2:]).float()
        th = jaccard(boxes, boxes).numpy()
        for i in range(th.shape[0]): th[i][i] = 0
        return np.max(th, axis=1)
    
    def dist(self, W, H, min=None, max=None):
        points = list((product(range(W),range(H))))
        dist = pairwise_distances(points, self.xy)        
        arr = dist.min(axis=1).reshape((W,H)).transpose()
        if min is not None: arr = arr.clip(min, None)
        if max is not None: arr = arr.clip(None, max)
        return arr
    
    def pdist(self, points):
        dist = pairwise_distances(points, self.xy)        
        return dist.min(axis=1)

    def vor(self, mind=None, maxd=None):
        points=self.xy
        vor = Voronoi(points)
        ab = np.array(points)[vor.ridge_points]
        vertices = np.concatenate([vor.vertices, np.array([[-1,-1]])],0)
        segments = vertices[np.array(vor.ridge_vertices)].astype('int32')
        x,y = segments[:,0],segments[:,1]
        a,b = ab[:,0],ab[:,1]
        #print("ab",ab.shape)
        mask = sign(x,y,a,b)
        m = (((x==-1)[:,0]) | ((y==-1)[:,0]))
        mab = ab[(mask|m)]
        #print("mab",mab.shape)
        #print("mab",mab)
        ma,mb = mab[:,0],mab[:,1]
        dist = ((ma-mb)**2).sum(axis=1) ** (0.5)
        if mind is not None:
            mask = dist>mind
            dist = dist[mask]
            mab = mab[mask]
        if maxd is not None:
            mask = dist<maxd
            dist = dist[mask]
            mab = mab[mask]
        xy = mab.mean(axis=1).astype('int32')
        
        #for i in range(dist.shape[0]):
        #    print(xy[i],mab[i,0],mab[i,1],dist[i])
        cl = np.ones((xy.shape[0],1))*-1
        cxy = np.concatenate([cl,xy],axis=1).astype('int32')
        return Ann(dets=cxy)
        
    def min_dist(self):
        xy = self.xy
        dist = squareform(pdist(xy))
        N = dist.shape[0]
        dist[range(N),range(N)] = dist.max() + 1
        return dist.min(0)

    def pnms(self, d):
        """
        supress points within distance d 
        based on heatmap 
        """
        if self.count == 0: return self
        h = self.conf
        xy = self.xy
        
        hxy = np.concatenate([h.reshape((-1,1)),xy],axis=1)
        keep = []
        while hxy.shape[0] > 0:
            keep.append(hxy[0])
            h,x,y = hxy[0]
            mask = (pairwise_distances([[x,y]], hxy[:,1:])>=d)[0]
            new_hxy = hxy[mask]
            hxy = new_hxy
        hxy = np.array(keep)
        #print(hxy)
        if len(hxy.shape) == 1: hxy = hxy.reshape(1,-1)
        h = hxy[:,0].copy()
        hxy[:,0] = -1
        ann = Ann(dets=hxy.astype('int32'))
        ann.conf = h
        return ann


    def pnms2(self, d):
        """
        supress points within distance d 
        based on heatmap 
        """
        if self.count == 0: return self
        h = self.conf
        dets = self.dets.copy()
        xy = self.xy
        
        hxy = np.concatenate([h.reshape((-1,1)),xy],axis=1)
        keep = []
        while hxy.shape[0] > 0:
            keep.append(dets[0])
            h,x,y = hxy[0]
            mask = (pairwise_distances([[x,y]], hxy[:,1:])>=d)[0]
            new_hxy = hxy[mask]
            hxy = new_hxy
            dets = dets[mask]
        dets = np.array(keep)
        ann = Ann(dets=dets)
        return ann
    
    
    def hmconf(self, YX):
        self = self.copy()
        ret = []
        for i,(x,y) in enumerate(self.xy):
            ret.append(1.0-YX[y,x]/255)
        
        
        self.conf = (np.array(ret)*100).astype('int32')
        return self
    
    def pr_curve(self):
        ann = self
        p = ann.conf/100
        y = ann.cl!=-1
        return list(zip(*precision_recall_curve(y,p)))
    
    
    
    def get_pr(self):
        ann = self
        p = ann.conf/100
        y = ann.cl!=-1
        p,r,t = precision_recall_curve(y,p)
        prt = {t:(p,r) for p,r,t in list(zip(p,r,t))}
        for i in range(6,10,1):
            c = i/10
            yield (int(c*100),prt[c])

        
    
def orientation( p,  q,  r):
    val = ((q[:,1] - p[:,1]) * (r[:,0] - q[:,0])) - ((q[:,0] - p[:,0]) * (r[:,1] - q[:,1]))  
    return val > 0

def sign(x,y,a,b):
    return  (orientation(x,y,a)!=orientation(x,y,b))&(orientation(a,b,x)!=orientation(a,b,y))
