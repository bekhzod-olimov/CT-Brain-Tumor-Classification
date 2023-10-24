# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn
from PIL import Image
from torchvision import transforms as T
from glob import glob
torch.manual_seed(2023)

class CustomDataset(Dataset):
    
    def __init__(self, root, n_cls, transformations = None):
        
        self.transformations = transformations
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/*/*")) if "jpg" in im_path]
        
        self.classes, count = {}, 0
        for idx, im_path in enumerate(self.im_paths):
            if len(self.classes) == n_cls: break
            class_name = self.get_class(im_path)
            if class_name not in self.classes: self.classes[class_name] = count; count += 1            
        
    def get_class(self, path): return os.path.dirname(path).split("/")[-1]
    
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path)
        gt = self.classes[self.get_class(im_path)]
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt

def get_dls(root, transformations, bs, n_cls, split = [0.8, 0.1, 0.1], ns = 4):
    
    ds = CustomDataset(root = root, transformations = transformations, n_cls = n_cls)
    ds_len = len(ds)
    tr_len = int(ds_len * split[0]); val_len = int(ds_len * split[1]); ts_len = ds_len - tr_len - val_len
    
    tr_ds, val_ds, ts_ds = random_split(ds, [tr_len, val_len, ts_len])
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, ds.classes