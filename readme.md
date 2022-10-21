# PyTorch StarDist
This repository contains PyTorch implementations for both StarDist 2D and StarDist 3D as describe in:

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
    conda activate pytorch_stardist
    cd stardist_tools_
    python setup.py install
    ```

## Training and Inference
the notebooks at `examples/3D` and `examples/2D` show in details how to perform training and inference.

## Notes
* This implementation doesn't support yet:
    - Multi-class classification of nuclei
    - Per patch inference on 2D images (Per patch inference on 3D images is implemented).
* The code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [stardist](https://github.com/stardist/stardist)
* The code in the folder `stardist_tools_` is from the [StarDist](https://github.com/stardist/stardist) repo.