from stardist_tools.csbdeep_utils import download_and_extract_zip_file

if __name__=="__main__":
    
    # Download the file file demo3D.zip that contains synthetic train and test images with associated ground truth labels.
    # source: https://github.com/stardist/stardist/blob/master/examples/3D/1_data.ipynb
    print("Dowloading dsb2018.zip")
    download_and_extract_zip_file(
        url       = 'https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir = 'datasets',
        verbose   = 1,
    )

    # Well annotated fluo images of the stage1_train images from the Kaggle 2018 Data Science Bowl
    # source: https://github.com/stardist/stardist/blob/master/examples/2D/1_data.ipynb
    print("\nDownloading demo3D.zip")
    download_and_extract_zip_file(
        url       = 'https://github.com/stardist/stardist/releases/download/0.3.0/demo3D.zip',
        targetdir = r'datasets/demo3d',
        verbose   = 1,
    )