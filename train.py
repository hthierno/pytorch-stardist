import sys
import warnings
import argparse
from pathlib import Path

from tqdm import tqdm
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stardist_tools import Rays_GoldenSpiral

from src.training import train
from src.data.stardist_dataset import get_train_val_dataloaders
from utils import seed_all, prepare_conf

from src.models.config import Config2D, Config3D
from src.models.stardist3d import StarDist3D
from src.models.stardist2d import StarDist2D

from evaluate import evaluate



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_conf", type=str, help="YAML configuration file.")
    return parser.parse_args()



def run():
    """
    Load configurations from the YAML file specified in the command
    Create the StarDist model
    Load the data
    Perform model training and thresholds optimization
    Perform evaluation on train and validation sets
    """
    args = get_arguments()

    with open(args.yaml_conf) as yconf:
        opt = yaml.safe_load(yconf)
    
    if len(opt['kernel_size'])==2:
        Config = Config2D
        StarDist = StarDist2D
    else:
        Config = Config3D
        StarDist = StarDist3D

    conf = Config( **opt )

    # Set random seed
    seed_all(conf.random_seed)

    # process the configuration variables
    opt = prepare_conf(conf)

    # Model instanciation
    model = StarDist(opt)
    
    # Raise a warning if the model's field of view is smaller than the median size of the objects.
    fov = np.array( [max(r) for r in model._compute_receptive_field()] )
    object_median_size = opt.extents

    print("Median object size".ljust(25), ":", object_median_size)
    print("Network field of veiw".ljust(25), ":", fov)

    if any(object_median_size > fov):
        warnings.warn("WARNING: median object size larger than field of view of the neural network.")

        

        k = input("Median object size larger than field of view of the neural network.\nDo you want to continue? Enter `no` to exit: ")
        if str(k).strip().lower() not in ['no', 'n']:
            sys.exit()


    # Loading data    
    rays = None
    if model.opt.is_3d:
        rays = Rays_GoldenSpiral(opt.n_rays, anisotropy=opt.anisotropy)

    train_dataloader, val_dataloader = get_train_val_dataloaders(opt, rays)

    train_dataloader, val_dataloader = get_train_val_dataloaders(opt, rays)
    

    total_nb_samples = len( train_dataloader.dataset ) + ( len(val_dataloader.dataset) if val_dataloader is not None else 0 )
    nb_samples_train = len(train_dataloader.dataset)
    nb_samples_val = total_nb_samples - nb_samples_train

    print("Total nb samples: ".ljust(40), total_nb_samples)
    print("Train nb samples: ".ljust(40), nb_samples_train)
    print("Val nb samples: ".ljust(40), nb_samples_val)

    print("Train augmentation".ljust(25), ":",  train_dataloader.dataset.opt.preprocess)
    print("Val augmentation".ljust(25), ":", val_dataloader.dataset.opt.preprocess)

    # Training
    train(model, train_dataloader, val_dataloader)

    # Threshold optimization
    conf.load_epoch="best"
    best_model = StarDist(conf)

    X, Y = val_dataloader.dataset.get_all_data()
    best_model.optimize_thresholds(X, Y)

    # Evaluation
    log_dir = Path(model.opt.log_dir) / f"{model.opt.name}"

    patch_size = None
    # Uncomment the next line To do inference per patch in order to reduce memory usage. NB: IMPLEMENTED ONLY FOR StarDist3D
    # patch_size=(32, 128, 128) # or patch_size = model.opt.patch_size

    # Evaluation on training set
    print("Evaluation on training set")
    X, Y = train_dataloader.dataset.get_all_data()

    Y_pred = [best_model.predict_instance(x, patch_size = patch_size)[0] for x in tqdm(X)]
    stats, fig = evaluate(Y_pred, Y)
    plt.savefig( log_dir / "acc_on_train_set.png" )
    plt.close(fig)
    stats = pd.DataFrame( stats )
    stats.to_csv(log_dir / "perf_on_train_set.csv", index=False)
    print(stats)



    # Evaluation on Validation set
    print("Evaluation on validation set")

    X, Y = val_dataloader.dataset.get_all_data()

    Y_pred = [best_model.predict_instance(x, patch_size = patch_size)[0] for x in tqdm(X)]
    stats, fig = evaluate(Y_pred, Y)
    plt.savefig( log_dir / "acc_on_val_set.png" )
    plt.close(fig)
    stats = pd.DataFrame( stats )
    stats.to_csv(log_dir / "perf_on_val_set.csv", index=False)
    print(stats)

    print(f"\n\nEvaluation scores saved at <{log_dir.absolute()}>")


if __name__=="__main__":
    run()