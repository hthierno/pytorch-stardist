import random

import numpy as np

import torch.nn.functional as F
import torch

import torchvision.transforms as transforms



def get_params(opt, size):
    d, h, w = size
    if "resize" in opt.preprocess:
        d,h,w = opt.resize_to

    new_h = h
    new_w = w
    new_d = d


    # Random crop params
    # ==================
    if "crop" in opt.preprocess:
        z = random.randint(0, np.maximum(0, new_d - opt.crop_size[0]))
        x = random.randint(0, np.maximum(0, new_h - opt.crop_size[1]))
        y = random.randint(0, np.maximum(0, new_w - opt.crop_size[2]))

        new_d = min(new_d, opt.crop_size[0])
        new_h = min(new_h, opt.crop_size[1])
        new_w = min(new_w, opt.crop_size[2])
        #new_d, new_h, new_w = opt.crop_size[:3]
    else:
        x=y=z=None
    
    d, h, w = new_d, new_h, new_w


    # Flip params
    # ===========
    flip = random.random() > 0.5
    flip_axis = [-2, -1, (-2, -1)][np.random.randint(3)] #np.random.choice([-2, -1, (-2, -1)])
    
    # Intensity change
    # ================
    change_intensity = random.random() > 0.5
    intensity_factor = None
    intensity_bias = None
    if "randintensity" in opt.preprocess:
        intensity_factor = np.random.uniform( *opt.intensity_factor_range )
        intensity_bias = np.random.uniform( *opt.intensity_bias_range )

    # random scale_params
    # ===================
    do_scale = random.random() > 0.5
    rs_z = None
    rs_x = None
    rs_y = None
    rscaled_size = np.array([d, h, w])


    if "randscale" in opt.preprocess:
        scale_limit = (0.9, 1.1) if not hasattr(opt, "scale_limit") else opt.scale_limit
        if isinstance(scale_limit, int) or isinstance(scale_limit, float):
            scale_limit = (1-scale_limit, 1+scale_limit)
        

        
        if scale_limit[0] != 1.0 or scale_limit[1] != 1.0:
            scale = np.random.uniform(*scale_limit)
            rscaled_size = [ int(d*scale), int( h*scale ), int( w*scale ) ]

            
            if scale > 1.:        
                target_d, target_h, target_w = rscaled_size # np.floor( np.array( rscaled_size ) * np.array( rscaled_size ) )
                rs_z = random.randint(0, np.maximum(0, target_d - d ))
                rs_x = random.randint(0, np.maximum(0, target_h - h ))
                rs_y = random.randint(0, np.maximum(0, target_w - w ))
        
        if not hasattr(opt, "no_scale_depth") or opt.no_scale_depth:
            rscaled_size[0] = d
            rs_z = 0
        

    return {'crop_pos': (z, x, y), 'flip': flip, 'flip_axis': flip_axis,
            
            "scale": do_scale,"random_scaled_size": rscaled_size, "random_scaled_crop_pos": (rs_z, rs_x, rs_y),

            "change_intensity": change_intensity, "intensity_factor":intensity_factor, "intensity_bias":intensity_bias
            
            }




def get_transforms(opt, params, is_mask=False):

    #method = Image.NEAREST if is_mask else cv2.INTER_CUBIC
    method = "nearest" if is_mask else "trilinear"
    
    transform_list = []

    if "resize" in opt.preprocess:
        target_size = opt.resize_to
        transform_list.append(transforms.Lambda( lambda vol: resize_volume(vol, target_size, method=method) ) )
    
    if "crop" in opt.preprocess:
        assert params is not None and "crop_pos" in params
        crop_size = opt.crop_size
        transform_list.append(transforms.Lambda( lambda vol: __crop(vol, params['crop_pos'], crop_size) ) )
    

    if "randscale" in opt.preprocess.lower():
        if params['scale']:
            target_size = params["random_scaled_size"]
            pos = params["random_scaled_crop_pos"]
            transform_list.append( transforms.Lambda( lambda vol: __scale_crop(vol, target_size, pos, method=method, padding_mode="reflect") ) )
    
    
    
    if "flip" in opt.preprocess:
        transform_list.append( transforms.Lambda( lambda vol: __flip(vol, params["flip"], axis=params["flip_axis"]) ) )


    if "randintensity" in opt.preprocess:
        if params["change_intensity"] and not is_mask:
            i_factor = params["intensity_factor"]
            i_bias = params["intensity_bias"]
            transform_list.append( transforms.Lambda( lambda vol: vol*i_factor + i_bias ) )
    # Normalization
    # transform_list.append( transforms.Lambda( lambda vol: __normalize(vol, is_mask=is_mask ) ) )

    return transforms.Compose(transform_list)




def resize_volume(volume, target_size, method="trilinear"):
    assert method in ("trilinear", "nearest")
    
    align_corners = None
    if method != "nearest":
        align_corners = True
    
    if len(target_size)==2:
        if volume.ndim > 2:
            target_size = (volume.shape[-3],) + tuple(target_size)
        else:
            target_size = (1,) + tuple(target_size)

    volume = torch.from_numpy( volume )#.unsqueeze(0).unsqueeze(0)

    ndim = volume.ndim
    
    for _ in range( 5 - ndim ):
        volume = volume.unsqueeze(0)

    res_volume = F.interpolate( volume, size=target_size, align_corners=align_corners, mode=method )
    
    #return res_volume.numpy()[0,0]

    for _ in range( 5 - ndim ):
        # Modified Here
        # This volume = volume[0]
        # By
        res_volume = res_volume[0]
        # ========================

    return res_volume.numpy()



def __crop(volume, pos, size):
    z1, x1, y1 = pos
    td, th, tw = size
    return volume[..., z1:z1+td, x1:x1+th, y1:y1+tw]


def __flip(volume, flip:bool, axis):
    if flip:
        return np.flip(volume, axis=axis)
    return volume



def __scale_crop(volume, target_size, pos, method, padding_mode="reflect"):
    if volume.shape[-2:] == target_size:
        return volume


    d, h, w = volume.shape

    tr_volume =  resize_volume(volume, target_size, method=method)

    if tr_volume.shape[-1] < volume.shape[-1]:
        # do padding
        # print("padding")
        dp = d - tr_volume.shape[-3]
        hp = h - tr_volume.shape[-2]
        wp = w - tr_volume.shape[-1]
        pad_width = [ 
            ( dp//2 , dp//2 + dp%2 ),
            ( hp//2, hp//2 + hp%2 ),
            ( wp//2, wp//2 + wp%2 )
        ]

        return np.pad( tr_volume, pad_width=pad_width, mode=padding_mode)
        
    else:
        # Crop
        # print("Cropping")
    
        z, x, y = pos
    
        
        pos = (z, x, y)
        size = volume.shape[-3:]

        #print(pos, tr_volume.shape, size)
        return __crop(tr_volume, pos, size)
