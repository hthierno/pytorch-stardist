import os
import warnings
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np


from stardist_tools.csbdeep_utils import normalize

from src.data.transforms import resize_volume
from src.data.utils import load_img, save_img, get_files

from src.models.utils import save_json
from src.models.config import Config2D, Config3D
from src.models.stardist_base import StarDistBase
from src.models.stardist3d import StarDist3D
from src.models.stardist2d import StarDist2D

from utils import seed_all



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_3d", action="store_true", default=False, help="Whether the model is 3D.")
    parser.add_argument("--make_image_binary", action="store_true", default=False, help="Convert the image to binary image. Used to connect object in 3D mask of 2d instances.")

    parser.add_argument("--name", type=str, default="", help="The model name.")
    parser.add_argument("--checkpoints_dir", type=str, default=r"./checkpoints", help="Epoch to load.")
    parser.add_argument("--load_epoch", type=str, default="best", help="Epoch to load.")

    parser.add_argument("--image_dir", type=str, required=True, help="Path of the directory containing the images.")
    parser.add_argument("--object_median_size", type=float, nargs='+', default=[], help="Estimation of the median size of object in the images in space separated format: [DEPTH] HIGH WIDTH ; ex: `15 15` in 2D and `8 15 15` in 3D . If given, will rescale the images so that the median object size is the same as the one in the dataset used to train the model.")
    parser.add_argument("--inference_patch_size", type=float, nargs='+', default=None, help="If given, will do inference on patches of size `patch_size` to reduce memory usage; The result will be the same as the one obtained by doing inference on the whole image.")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Whether to use GPU.")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Whether to use Automatic Mixed Precision.")

    parser.add_argument("--results_dir", type=str, default=r"./results" ,help="Path of the directory containing the images.")


    return parser.parse_args()

def predict(model: StarDistBase, image_paths, object_median_size=None, normalize_channel="independently", patch_size=None, make_image_binary=False):
    
    assert normalize_channel in ("independently", "jointly", "none"), normalize_channel
    
    model_is_3d = len(model.opt.kernel_size)==3

    if normalize_channel != "none":
        if model_is_3d:
            axis_norm = (-1, -2, -3) if normalize_channel=="independently" else (-1, -2, -3, -4)
        else:
            axis_norm = (-1, -2) if normalize_channel=="independently" else (-1, -2, -3)

    scale = None
    if object_median_size is not None and len(object_median_size) > 0:
        scale = np.array( model.opt.extents ) / object_median_size
        print("\nImage will be scaled for model inference with scale =" , scale)

    if not patch_size: # == [] or is None...
        patch_size = None

    for image_path in tqdm(image_paths):
        image = load_img(image_path).squeeze().astype("float32")
        if model.opt.n_channel_in == 1:
            image = image[np.newaxis]
        
        if make_image_binary:
            image = 255. * (image!=0)

        image = normalize(image,1,99.8,axis=axis_norm)


        if scale is not None:
            orig_size = np.array( image.shape[-len(scale):] ).astype(int)
            target_size = ( orig_size * scale ).astype(int)
            image = resize_volume( image, target_size=tuple(target_size), method="trilinear" )

        if image.ndim==4 and not model_is_3d:
            # make predictions independently for each slice
            depth = image.shape[1]
            pred_mask = np.stack( [model.predict_instance(image[:, d], patch_size=patch_size)[0] for d in range(depth) ], axis=0 )
            

        else:
            pred_mask = model.predict_instance(image, patch_size=patch_size)[0]

        if scale is not None:
            # scaling back to the initial size
            image = resize_volume( image, target_size=tuple(orig_size), method="trilinear" )
            pred_mask = resize_volume( pred_mask.astype("float32"), target_size=tuple(orig_size), method="nearest" ).astype( pred_mask.dtype )

        yield image_path, image, pred_mask



        

def run(predict=predict):
    args = get_arguments()

    # Check that the size of `median_object_size` is consistent with `is_3d`
    tmp_ndim = len(args.object_median_size)
    if tmp_ndim > 0:
        if args.is_3d:
            assert tmp_ndim == 3, f"len(object_median_size)={tmp_ndim} but should be = 3"
        elif tmp_ndim == 2:
            warnings.warn(f"len(object_median_size)={tmp_ndim} whereas model is `is_3d`=False. If the image is in 3D, the inference will be performed independently on each slice.")
    else:
        args.object_median_size = None
    del tmp_ndim

    assert os.path.isdir(args.image_dir), f"The image directory {args.image_dir} does not exist"
    image_dir = Path(args.image_dir)
    image_paths = get_files( image_dir )


    if args.is_3d:
        Config = Config3D
        StarDist = StarDist3D
    else:
        Config = Config2D
        StarDist = StarDist2D


    opt = Config( allow_new_params=True, **vars(args) )
    opt.update_params(isTrain=False)

    # Set random seed
    seed_all(opt.random_seed)

    # Load the model
    model = StarDist(opt)


    print("\n Starting Inference")
    dest_dir = Path( args.results_dir ) / f"model_{model.opt.name}_{ 'epoch_' + str(opt.load_epoch) }"

    os.makedirs(dest_dir, exist_ok=True)
    dest_img_dir = dest_dir/"images"
    dest_pimg_dir = dest_dir/"processed_images"
    dest_mask_dir = dest_dir/"predicted_masks"
    
    dest_img_dir.mkdir(exist_ok=True)
    dest_pimg_dir.mkdir(exist_ok=True)
    dest_mask_dir.mkdir(exist_ok=True)

    print("Predictions will be saved at: ", dest_dir.absolute())


    for (image_path, image, mask) in predict(model, image_paths, object_median_size=args.object_median_size, patch_size=args.inference_patch_size, make_image_binary=args.make_image_binary):

        sub_dirs = image_path.relative_to( image_dir ).parent

        target_pimage_filepath = (dest_pimg_dir / sub_dirs) / f"{image_path.stem}.tif"
        target_image_filepath = (dest_img_dir / sub_dirs) / image_path.name
        target_mask_filepath = (dest_mask_dir / sub_dirs) / image_path.name

        os.makedirs( target_pimage_filepath.parent, exist_ok=True )
        os.makedirs( target_image_filepath.parent, exist_ok=True )
        os.makedirs( target_mask_filepath.parent, exist_ok=True )
    
        save_img(target_pimage_filepath, image)
        save_img(target_image_filepath, load_img(image_path))
        save_img(target_mask_filepath, mask)

    
    save_json( dest_dir / "info_prediction_args.json", vars(args) )

if __name__=="__main__":
    run()