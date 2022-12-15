import os
import random
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch

from stardist_tools import calculate_extents, Rays_GoldenSpiral, random_label_cmap
from src.data.stardist_dataset import get_train_val_dataloaders
from src.models.config import ConfigBase



def seed_all(seed):
    """Utility function to set seed across all pytorch process for repeatable experiment
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Utility function to set random seed for Pytorch DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def makedirs(path):
    """
    create a leaf directory and all intermediate ones if they don't aleardy exist

    path: str
        path to directory to create.
        if it is a path to a file, will consider the path to its direct parent directory
    """
    path = Path(path)
    if path.suffix:
        path = path.parent
    if not os.path.exists(path):
        print(f"The path <{path}> doesn't exists. It will be created!")
        os.makedirs(path)



def prepare_conf(opt:ConfigBase):

    """
    Add information to the configuration parameters.
    Attribute added/updated:
        - extents: tuple
            median size of object in the dataset
        - anisotropy: tuple
            anisotropy of object in the dataset. Computed as max(extents) / extents
        - grid: tuple
            if `grid`="auto", 
        - resnet_n_downs: tuple
            Number of downsampling in the resnet network. set to the `grid` parameter
    
    opt: ConfigBase
        Experiment configuration.

    """

    opt.n_channel = opt.n_channel_in
    opt.is_3d = len(opt.kernel_size)==3

    if not opt.is_3d:

        # Converting some 2d params to equivalent 3d params. example: crop_size = [256,256] -> [1, 256, 256]
        for attr in ('crop_size', 'resize_to'):
            if not hasattr(opt, attr):
                warnings.warn(f"attribue <{attr}> is not in configurations")
                continue

            value = opt.__getattribute__(attr)
            value = [1] + value
            opt.__setattr__(attr, value)


    if not hasattr(opt, "use_opencl") :
        opt.use_opencl = opt.use_gpu

    rays = Rays_GoldenSpiral(opt.n_rays)
    anisotropy = opt.anisotropy
    print()

    print(" === Computing extents...")

    train_dataloader, val_dataloader = get_train_val_dataloaders(opt, rays)
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset

    images, masks = train_dataset.get_all_data()

    val_images, val_masks = val_dataset.get_all_data()
    images += val_images
    val_masks += val_masks


    extents = calculate_extents(masks)
    opt.extents = list(extents)


    if anisotropy == "auto":
        print(" === Computing anisotropy...")
        anisotropy = tuple( np.max(extents) / extents )
        opt.anisotropy = anisotropy
    
    print(' === Empirical anisotropy of labeled objects = %s' % str(anisotropy))

    grid = opt.grid
    if grid == "auto":
        grid = tuple( np.round( max(anisotropy) / a ).astype(int) for a in anisotropy )  #tuple(1 if a > 1.5 else 2 for a in anisotropy)
        grid = tuple(make_power_of_2(grid))
        if max(grid)==1:
            grid = (2,) * len(grid)
        print(" === 'grid' set to", grid)
        opt.grid = grid
    
    opt.resnet_n_downs = opt.grid

    print()

    return opt

def make_power_of_2(arr):
    return 2 ** np.ceil( np.log2(arr) ).astype(int)


lbl_cmap = random_label_cmap()

def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

def plot_img_label2d(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()