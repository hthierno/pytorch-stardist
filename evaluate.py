import os
import argparse
from pathlib import Path

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from stardist_tools.matching import matching_dataset

from src.data.utils import load_img, get_files


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_mask_dir", type=str, required=True, help="Path of the directory containing the ground-truth masks.")
    parser.add_argument("--predicted_mask_dir", type=str, default=None, help="Path of the directory containing the predicted masks.")

    parser.add_argument("--in_2d", action="store_true", default=False, help="If issued, will perform evaluation in 2D even if the mask is 3D.")


    return parser.parse_args()


def evaluate(Y_pred, Y):
    """
    Compute evaluation metrics

    parameters
    ----------
    Y_pred: list of predicted instance segmentation map
    Y: list of ground-truth corresponding to the predictions Y_pred 
    """

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()

    return stats, fig



def run():
    args = get_arguments()

    
    assert os.path.isdir(args.real_mask_dir), f"The directory of ground truth masks {args.real_mask_dir} does not exist."
    assert os.path.isdir(args.predicted_mask_dir), f"The directory of predicted masks {args.predicted_mask_dir} does not exist."
    
    real_mask_paths = get_files( Path(args.real_mask_dir) )
    real_mask_paths.sort()
    
    predicted_mask_paths = get_files( Path(args.predicted_mask_dir) )
    predicted_mask_paths.sort()

    n_real = len(real_mask_paths)
    n_predicted = len(predicted_mask_paths)
    assert n_real == n_predicted, f"number of real masks ({n_real}) is different of the number of predicted_mask ({n_predicted})."

    real_masks = []
    predicted_masks = []

    print("Loading masks")

    for real_mask_filepath, predicted_mask_filepath in zip(real_mask_paths, predicted_mask_paths):
        
        real_mask = load_img(real_mask_filepath).squeeze()
        predicted_mask = load_img(predicted_mask_filepath).squeeze()

        assert ( real_mask == real_mask.astype(predicted_mask.dtype) ).all(), (real_mask.dtype, predicted_mask.dtype)
        real_mask = real_mask.astype(predicted_mask.dtype)

        assert real_mask.shape == predicted_mask.shape, f"shape of real mask <{real_mask_filepath}> - {real_mask.shape} != {predicted_mask.shape} - shape of corresponding prediction <{predicted_mask_filepath}>"

        assert real_mask.ndim in (2, 3), real_mask.shape

        if args.in_2d and real_mask.ndim == 3:
            real_mask = [ mask_slice for mask_slice in real_mask ]
            predicted_mask = [ mask_slice for mask_slice in predicted_mask ]
        else:
            real_mask = [real_mask]
            predicted_mask = [predicted_mask]
        
        real_masks += real_mask
        predicted_masks += predicted_mask
    
    print("Performing Evaluation")

    predicted_mask_dir = Path( args.predicted_mask_dir )
    log_dir = predicted_mask_dir.parent / f"eval_metrics_{predicted_mask_dir.name}{'_in_2d' if args.in_2d else ''}"
    log_dir.mkdir(exist_ok=True)


    stats, fig = evaluate(predicted_masks, real_masks)
    plt.savefig( log_dir / "acc.png" )
    plt.close(fig)
    stats = pd.DataFrame( stats )
    stats.to_csv(log_dir / "perf.csv", index=False)
    print(stats)

    print(f"\n\nEvaluation scores saved at <{log_dir.absolute()}>")


if __name__=="__main__":
    run()