import functools
import warnings
import functools
from pathlib import Path
import os
from collections import defaultdict
import time

import json
import numpy as np

import torch
from torch.optim import lr_scheduler


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data

def makedirs(path):
    path = Path(path)
    if path.suffix:
        path = path.parent
    if not os.path.exists(path):
        print(f"The path <{path}> doesn't exists. It will be created!")
        os.makedirs(path)

def with_no_grad(f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return g

class TimeTracker:
    """
    class to track execution time
    """
    def __init__(self):
        self.dict_start_end = defaultdict( list )
        self.phases = defaultdict( list )
    
    def tic(self, phase:str):
        self.dict_start_end[phase].append( [time.time(), None] )
    
    def tac(self, phase:str):
        end=time.time()
        start = self.dict_start_end[phase][-1][0]
        self.dict_start_end[phase][-1][-1] = end
        self.phases[phase].append( end-start )
    
    def get_duration(self, phase):
        return self.phases[phase]
    def __getitem__(self, phase):
        return self.phases[phase]


def update_lr(optimizer, scheduler, opt, metric=None):
    """
        Update the learning rate
    parameters
    ----------
    opt: ConfigBase
        object with the model configurations
    """
    if opt.lr_policy == "plateau":
        assert metric is not None, "lr_policy='plateau' but metric is None"

    old_lr = optimizer.param_groups[0]['lr']
    if opt.lr_policy == 'plateau':
        scheduler.step(metric)
    else:
        scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

    return lr

def get_scheduler(optimizer, opt, init_lr=None):
    # modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "none":
        def lambda_rule(epoch):
            return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, opt.lr_linear_n_epochs - epoch) / float(opt.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'linear_decay':
        def lambda_rule(epoch):
            lr_l = 1 - epoch / float(opt.n_epochs + 1)
            if lr_l <= 0:
                warnings.warn("learning rate negative. current epoch number is higher than the initially total epopchs setted.") 
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_n_epochs, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_plateau_factor, threshold=opt.lr_plateau_threshold, patience=opt.lr_plateau_patience, min_lr=opt.min_lr)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



def _make_grid_divisible(grid, size, name=None, verbose=True):
    size = np.array(size)
    grid = np.array(grid)

    if (size % grid == 0).all():
        return size
    _size = size
    size = np.ceil(size / grid) * grid
    size = size.astype(int)
    if bool(verbose):
        print(f"{verbose if isinstance(verbose,str) else ''}increasing '{'value' if name is None else name}' from {_size} to {size} to be evenly divisible by {grid} (grid)", flush=True)
    assert (size % grid == 0).all()
    return size

def _make_power_of_2(arr, n=1):
    n=n-1
    return ( np.array(arr) // 2**n ) * 2**n 
def _is_power_of_2(arr, n=1):
    n=n-1
    arr = np.array(arr)
    q = arr / 2**n
    return (q.astype(int) == q) #| (arr==1) 

class Block3D:
    """
    This class allows to decompose a 3D volume into overlapping patches to process them externally and to construct a new volume from the processed patches.
    Used by `StarDist3D.predict_big` for per patch prediction.

    When iterating over an instance (e.g. with a "for loop"), it returns a patch of the volume and an identifier.
    You can then do whatever processing you want on that patch, then call the `set_target_patch` method which will
    place the processed patch in a volume at a position that matches its position in the original volume.
    """
    def __init__( self, volume, n_channel_target=None, grid=None, patch_size=None, context = None ):
        
        ndim = volume.ndim - 1 # -1 for channel
        assert len(grid) == ndim, (grid, ndim)
        assert len(patch_size) == ndim, (patch_size, ndim)
        assert len(context) == ndim, (context, ndim)

        self.volume = volume


        if n_channel_target == None:
            n_channel_target = volume.shape[0]

        self.patch_size = np.array( [
            patch_size,
            volume.shape[1:] #volume.shape[-len(patch_size):]
        ] ).min(axis=0)

        grid = np.array(grid)
        assert _is_power_of_2( grid ).all(), grid
        assert _is_power_of_2( patch_size, n=grid ).all(), patch_size
        assert _is_power_of_2( context, n=grid ).all(), context

        self.grid = grid
        self.patch_size = np.array(patch_size)
        self.context = np.array(context)

        target_shape = [n_channel_target] + [ int((a % s)!=0) +  a // s for a, s in zip( volume.shape[1:], grid )]
        self._target_volume = np.zeros( target_shape )

        self.slices = None
        # Computing slices
        self._compute_patch_slices()

    def get_target_volume(self):
        if len(self.slices)>0:
            warnings.warn( f"Some patches are still not processed!" )
        return self._target_volume

    def __iter__(self):
        self.list_idx = list(self.slices.keys())
        self.i = 0

        assert len( self.list_idx ) > 0, f"All blocks already processed"
        return self
    
    def __next__(self):
        if self.i >= len(self.list_idx):
            raise StopIteration
        

        idx = self.list_idx[self.i]
        
        block_info = self.slices[idx]
        slice_z, slice_x, slice_y = block_info["src_slices"]

        self.i+=1

        return idx, self.volume[..., slice_z, slice_x, slice_y]
    
    def set_target_patch( self, idx, patch ):
        assert idx in self.slices.keys(), f"patch of idx={idx} has already been set"

        #assert patch.shape[-3:] == tuple(self.patch_size), f"The size of the given patch {patch.shape}[-3:] != {tuple(self.patch_size)}"


        block_info = self.slices[idx]

        #target_patch_start = block_info['patch_start'] // self.grid + (block_info['patch_start'] % self.grid)==0).astype(int)
        #target_slice_z, target_slice_y, target_slice_y = block_info["tgt_slices"]

        src_tgt_slices = block_info["src_tgt_slices"]
        tgt_slices = block_info["tgt_slices"]



        self._target_volume[..., src_tgt_slices[0], src_tgt_slices[1], src_tgt_slices[2]] = patch[..., tgt_slices[0], tgt_slices[1], tgt_slices[2]]

        del self.slices[idx]


    def _compute_patch_slices(self):

        slices = dict()
        volume = self.volume

        d, h, w = self.patch_size
        depth, height, width = volume.shape[-3:]
        offset_d, offset_h, offset_w = np.array( self.patch_size )

        ctx_z, ctx_x, ctx_y = self.context

        # Define patch starts (below-left-top)
        z = np.arange(0, depth, offset_d)
        x = np.arange(0, height, offset_h)
        y = np.arange(0, width, offset_w)

        z  = z[ (depth - z) > d ]
        x  = x[ (height - x) > h ]
        y  = y[ (width - y) > w ]

        z = np.append( z, depth - d )
        x = np.append( x, height - h )
        y = np.append( y, width - w )

        z = np.clip( z, 0, None )
        x = np.clip( x, 0, None )
        y = np.clip( y, 0, None )

        # print(z)
        # print(x)
        # print(y)

        
        # context sizes
        ## ctx_axis_a: left context size for axis
        ## ctx_axis_b: right context size for axis
        ctx_z_b = np.full_like(z, ctx_z)
        ctx_z_a = np.full_like(z, ctx_z)

        ctx_x_b = np.full_like(x, ctx_x)
        ctx_x_a = np.full_like(x, ctx_x)

        ctx_y_b = np.full_like(y, ctx_y)
        ctx_y_a = np.full_like(y, ctx_y)

        # Reduce context sizes that are too large
        ctx_z_b[ (-ctx_z_b + z) < 0 ] = 0
        ctx_x_b[ (-ctx_x_b + x) < 0 ] = 0
        ctx_y_b[ (-ctx_y_b + y) < 0 ] = 0

        ctx_z_a[ (ctx_z_a + z + d) > depth ] = 0
        ctx_x_a[ (ctx_x_a + x + h) > height ] = 0
        ctx_y_a[ (ctx_y_a + y + w) > width ] = 0

        idx = 0
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):

                    xi = x[i]
                    yj = y[j]
                    zk = z[k]

                    ctx_b_zk = ctx_z + (k == len(z)-1) * (depth%2)
                    ctx_b_zk = ctx_b_zk if (zk - ctx_b_zk) >= 0 else _make_power_of_2(zk, self.grid[0])

                    ctx_b_xi = ctx_x + (i == len(x)-1) * (height%2)
                    ctx_b_xi = ctx_b_xi if (xi - ctx_b_xi) >= 0 else _make_power_of_2(xi, self.grid[1])

                    ctx_b_yj = ctx_y + (j == len(y)-1) * (width%2)
                    ctx_b_yj = ctx_b_yj if (yj - ctx_b_yj) >= 0 else _make_power_of_2(yj, self.grid[2])


                    ctx_b = np.array([ctx_b_zk, ctx_b_xi, ctx_b_yj])
                    
                    target_zk, target_xi, target_yj = ( (ctx_b % self.grid)!=0 ).astype(int) + ctx_b // self.grid
                    
    

                    slice_z = slice(zk-ctx_b_zk, zk+d+ctx_z)
                    slice_x = slice(xi-ctx_b_xi, xi+h+ctx_x)
                    slice_y = slice(yj-ctx_b_yj, yj+w+ctx_y)

                    target_slice_z = slice(target_zk, target_zk + d // self.grid[0] )
                    target_slice_x = slice(target_xi, target_xi + h // self.grid[1] )
                    target_slice_y = slice(target_yj, target_yj + w // self.grid[2] )


                    src_patch_start = np.array( [ zk, xi, yj ] )
                    src_target_z, src_target_x, src_target_y = ( (src_patch_start % self.grid)!=0 ).astype(int) + src_patch_start // self.grid
                    
                    src_target_slice_z = slice( src_target_z, src_target_z + d // self.grid[0] )
                    src_target_slice_x = slice( src_target_x, src_target_x + h // self.grid[1] )
                    src_target_slice_y = slice( src_target_y, src_target_y + w // self.grid[2] )


                    slices[idx] = {
                        "src_slices": [ slice_z, slice_x, slice_y ],
                        "src_tgt_slices": [ src_target_slice_z, src_target_slice_x, src_target_slice_y ],
                        "tgt_slices": [ target_slice_z, target_slice_x, target_slice_y ]
                        }

                    idx += 1
        
        self.slices = slices

