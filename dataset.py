# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T; from glob import glob
torch.manual_seed(2023)

class CustomDataset(Dataset):

    """
    
    This class gets several parameters and return custom dataset.

    Parameters:

        root              - path to data, str;
        n_cls             - number of classes in the dataset, int;
        transformations   - transformations to be applied, torchvision transforms object.
        
    """
    
    def __init__(self, root, n_cls, transformations = None):
        
        # Get the transformations
        self.transformations = transformations
        # Get the image files list from the directory
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/*/*")) if "jpg" in im_path]

        # Get classes and their counts
        self.classes, count = {}, 0
        # Go through every path 
        for idx, im_path in enumerate(self.im_paths):
            if len(self.classes) == n_cls: break
            # Get class name based on the path
            class_name = self.get_class(im_path)
            # Add count
            if class_name not in self.classes: self.classes[class_name] = count; count += 1            
        
    # Function to get class name based on the path
    def get_class(self, path): return os.path.dirname(path).split("/")[-1]
    
    # Function to get the length of the dataset
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):

        """
        
        This function gets an index and returns image and label pair.

        Parameter:

            idx       - index, int;

        Outputs:

            im        - image, tensor;
            gt        - class, int.
        
        """
        
        # Get an image path
        im_path = self.im_paths[idx]
        # Read an image based on the path
        im = Image.open(im_path)
        # Get label
        gt = self.classes[self.get_class(im_path)]
        # Apply transformations
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt

def get_dls(root, transformations, bs, n_cls, split = [0.8, 0.1, 0.1], ns = 4):

    """
    
    This function gets several parameters and returns train, validation, and test dataloaders.

    Parameters:

        root              - path to data, str;
        transformations   - transformations to be applied, torchvision transforms object;
        bs                - mini batch size, int;
        n_cls             - number of classes in the dataset, int;
        split             - split for train, validation, and test data, list -> float;
        ns                - number of workers, int.
        
    """
    
    # Get dataset
    ds = CustomDataset(root = root, transformations = transformations, n_cls = n_cls)
    # Get the dataset length
    ds_len = len(ds)
    # Get the train, validation, and test set lentghs
    tr_len = int(ds_len * split[0]); val_len = int(ds_len * split[1]); ts_len = ds_len - tr_len - val_len
    # Split the data based on the lengths
    tr_ds, val_ds, ts_ds = random_split(ds, [tr_len, val_len, ts_len])
    # Create train, validation, and test dataloaders
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, ds.classes
