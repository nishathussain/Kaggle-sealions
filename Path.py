
class Path(object):
    def __init__(self, paths):
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
    
    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = os.path.join(*self.source_paths[name]).format(**kwargs)
        return path        
    
    def dirpath(self, name, **kwargs):
            path = os.path.join(*self.source_paths[name][:-1]).format(**kwargs)
            return path

    def mkpath(self, name, **kwargs):
        path = self.dirpath(name, **kwargs)
        os.makedirs(path, exist_ok=True)
