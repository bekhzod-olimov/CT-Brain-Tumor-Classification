# Import library
from torchvision import transforms as T

def get_tfs(im_size = (224, 224), imagenet_normalization = True):

    """
    
    This function gets several parameters and returns train and test transformations.

    Parameters:

        im_size                 - image dimension to be resized, tuple -> int;
        imagenet_normalization  - whether or not to use ImageNet normalization, bool.
            
    """
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return T.Compose([T.Resize(im_size), T.ToTensor(), T.Normalize(mean = mean, std = std)]) if imagenet_normalization else T.Compose([T.Resize(im_size), T.ToTensor()])
