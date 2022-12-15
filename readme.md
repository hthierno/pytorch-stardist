# PyTorch StarDist
This repository contains PyTorch implementations for both StarDist 2D and StarDist 3D as described in:

- Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.  
[*Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy*](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf).  
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020


## Installation

You should have a C++ compiler installed as this code relies on C/C++ extensions that need to be compiled. This code has been tested with [Build Tools for Visual Studio](https://visualstudio.microsoft.com/fr/downloads/#build-tools-for-visual-studio-2022) on Windows and GCC on linux.

Follow this step to install pytorch stardist:

1. Download the repo
2. Create a conda environment using the file `environment.yml`:
    `conda env create --file environment.yml`
3. Activate the environment and install the package `stardist_tools`:
    ```
    conda activate pytorch-stardist
    cd stardist_tools_
    python setup.py install
    ```

## Training and Inference
### Notebooks
The notebooks at `examples/3D` and `examples/2D` show in details how to perform training and inference.

### Command line scipts
You can also use command line script.

Let's download some data for the demonstration:

```
python download_demo_data.py
```

#### Training
You need a YAML file containing the training configurations to run the `train.py` script . Check `confs\dsb2018_2d.yml` and `confs\demo_3d.yml` for examples of configuration files.

Run the following command to train the model with the configurations in `confs\dsb2018.yml`: 

```
python train.py --yaml_conf .\confs\dsb2018.yml
```

It will train a starDist2D model on a subsample of the Data Science Bowl 2018 dataset. 

#### Inference
We can perform predictions with the trained model using:
```
python predict.py --name dsb2018 --checkpoints_dir .\checkpoints --image_dir .\datasets\dsb2018\test\images --result_dir .\datasets\dsb2018\preds --use_gpu --use_amp
```

`--name dsb2018` indicates the name of the experiment given in the YAML configuration file used for the training.

<p align="center">
  <img src="http://https://github.com/hthierno/pytorch-stardist/images/example_preds.png", alt="example_of_prediction" />
</p>

## Notes
* The code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [stardist](https://github.com/stardist/stardist)
* The code in the folder `stardist_tools_` is from the [StarDist](https://github.com/stardist/stardist) repo.