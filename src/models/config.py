class ConfigBase:

    """
        Configuration for a StarDist model.

        Parameters
        ----------
        data_dir: str or None
            path to data directory with the following structure:
            
            data_dir
                |train
                |----|images
                |----|masks
                |val [Optional]
                |----|images
                |----|masks

            if the `val` directory is absent, the data in the `train` folder will be split.
            
            
        
        patch_size: tuple
            size of image to crop from original images.
        load_epoch: int or 'best' or None
            if not None, will load state corresponding to epoch `load_epoch`
        
        Attributes
        ----------
        name: str
            Name to give to the model
        random_seed: int
            random seed to use for reproducibility

        log_dir: str
            directory path where to save the logs
        checkpoint_dir: str
            directory where to save model states

        # ========================= dataset ==================================
        val_size: float - default 0.15
            Fraction (0...1) of data from the `train` folder to use as validation set when the `val` folder doesn't exist 
        n_rays: int
            Number of rays to use in in the star-convex representation of nuclei shape  
        foreground_prob: float between 0 and 1
            Fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels.
        cache_sample_ind: bool
            whether to keep in RAM indices of valid patches
        cache_data: bool
            whether to keep in RAM training data
        
        batch_size: int
            size of batches
        num_workers: int
            Number of subprocesses to use for data training.
        
        preprocess: str
            type of augmentation to do on training data.
            available augmentations: none|flip|randintensity|randscale|resize
            you can use muliple augmentation, for example for radnom fliping and random intensity scaling, set: `flip_randintensity`  
        preprocess_val: str
            same as preprocess but on validation data
        intensity_factor_range: (float, float):
            range from which to sample weight to multiply image intensities.
            Associated to `randintensity` augmentation.             
        intensity_bias_range: (float, float)
            range from which to sample bias to add to image intentsities.
            Associated to `randintensity` augmentation. 
        scale_limit: (float, float):
            range from which to sample scale to apply the image.
            Associated to `randscale` augmentation. 
        resize_to: tuple
            size to which to resize each image.

        #======================================================================

        # ========================= Training ==================================

        use_gpu: bool
            whether to use GPU
        use_amp: bool
            whether to use Automatic Mixed Precision
        isTrain: bool
            whether to initialize model in traning mode (set optimizers, schedulers ...)
        evaluate: bool
            whether to perform evaluation during traning.


        n_epochs: int
            Number of training epochs
        self.n_steps_per_epoch:
            Number of weights updates per epoch

        lambda_prob: flaot
            Weight for probablitly loss
        lambda_dist:
            Weight for distance loss
        lambda_reg: float
            Regularizer to encourage distance predictions on background regions to be 0.

        start_saving_best_after_epoch: int
            Epoch after which to start to save the best model


        #======================================================================



        # ========================= Networks configurations ==================
        grid: str
            Subsampling factors (must be powers of 2) for each of the axes.
            Model will predict on a subsampled grid for increased efficiency and larger field of view.

        n_channel_in: int
            Number of channel of images
        kernel_size: tuple
            Kernel size to use in neural network

        resnet_n_blocks: int
            Number of ResNet blocks to use
        n_filter_of_conv_after_resnet: int
            Number of filter in the convolution layer before the final prediction layer.
        resnet_n_filter_base: int
            Number of filter to use in the first convolution layer
        resnet_n_conv_per_block: int
            Number of convolution layers to use in each ResNet block.

        #======================================================================


        # ========================= Optimizers ================================
        lr: float
            Learning rate
        lr_policy: str
            learning rate scheduler policy. 
            Possible values: 
                - "none" -> keep the same learning rate for all epochs
                - "plateau" -> Pytorch ReduceLROnPlateau scheduler
                - "linear_decay" -> linearly decay learning rate from `lr` to 0
                - "linear" -> linearly increase  learning rate from 0 to `lr` during the first `lr_linear_n_epochs` and use `lr` for the remaining epochs
                - "step" -> reduce learning rate by 10 every `lr_step_n_epochs`
                - "cosine" -> Pytorch CosineAnnealingLR

        | Parameter for ReduceLROnPlateau used when `lr_policy` = "plateau"
        ------------------------------------------------------------------
        | lr_plateau_factor: float
        | lr_plateau_threshold: float
        | lr_plateau_patience: float
        | min_lr:float
        ------------------------------------------------------------------

        self.lr_linear_n_epochs : int
            See `lr_policy` when `lr_policy`="linear"
        self.lr_step_n_epochs: int
            See `lr_policy` when `lr_policy`="step"
        self.T_max: int
            T_max parameter of Pytorch CosineAnnealingLR.
            Used when `lr_policy` = "cosine

    """

    def update_params(self, allow_new_params=False, **kwargs):
        return self._update_params_from_dict(allow_new_params, kwargs)
    
    def _update_params_from_dict(self, allow_new_params, param_dict):
        if not allow_new_params:
            attr_new = []
            for k in param_dict:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in param_dict:
            setattr(self, k, param_dict[k])
    
    def get_params_value(self):
        return self.__dict__


class Config2D(ConfigBase):
    def __init__(
        self, 
        name,
        data_dir=None,
        patch_size = [256, 256],
        load_epoch=None,
        **kwargs
    ):
        super().__init__()

        self.delay_hours                    = 0.0

        self.name                           = name #'c_elegans_orig'
        self.random_seed                    = 42


        self.log_dir                        =  './logs/'
        self.checkpoints_dir                =  "./checkpoints"
        self. result_dir                     =  "./results"

        # ========================= dataset ==================================

        self.data_dir                       = data_dir #'datasets/c_elegans_processed'
        self.val_size                       =  0.15
        self.n_rays                         = 32
        self.foreground_prob                = 0.9
        self.n_classes                      = None # non None value (multiclass) not supported yet
        self.patch_size                     = patch_size
        self.cache_sample_ind               = True
        self.cache_data                     = True

        self.batch_size                     = 4
        self.num_workers                    = 0

        self.preprocess                     = "none"
        self.preprocess_val                 = "none"
        self.intensity_factor_range         = [0.6, 2.]
        self.intensity_bias_range           = [-0.2, 0.2]
        self.scale_limit                    = [1., 1.1]
        self.resize_to                      = [5, 286, 286]
        self.crop_size                      = [1, 256, 256]

        #======================================================================





        # ========================= Training ==================================

        self.use_gpu                        = True
        self.use_amp                        = True
        self.isTrain                        = True 
        self.evaluate                       = True #True
        #self.gpu_ids                       = [0]
        #self.continue_train                = False


        self.load_epoch                     = load_epoch
        self.n_epochs                       = 400
        self.n_steps_per_epoch              = 100


        self.lambda_prob                    = 1.
        self.lambda_dist                    = 0.2
        self.lambda_reg                     = 0.0001
        self.lambda_prob_class              = 1.

        self.save_epoch_freq                = 50
        self.start_saving_best_after_epoch  = 50 


        #======================================================================



        # ========================= Networks configurations ==================
        self.init_type                      = "normal"
        self.init_gain                      = 0.02

        self.backbone                       = "resnet"
        self.grid                           = "auto"
        self.anisotropy                     = "auto"

        self.n_channel_in                   = 1
        self.kernel_size                    = [3, 3]
        self.resnet_n_blocks                = 3
        self.resnet_n_downs                 = None  # WILL BE SET to 'grid' in the code
        self.n_filter_of_conv_after_resnet  = 128
        self.resnet_n_filter_base           = 32
        self.resnet_n_conv_per_block        = 3
        self.use_batch_norm                 = False

        #======================================================================


        # ========================= Optimizers ================================
        self.lr                             = 0.0003
        self.beta1                          =  0.9
        self.beta2                          =  0.999

        self.lr_policy                      = "plateau"
        self.lr_plateau_factor              = 0.5
        self.lr_plateau_threshold           = 0.0000001
        self.lr_plateau_patience            = 40
        self.min_lr                         = 1e-6


        self.lr_linear_n_epochs             = 100
        self.lr_decay_iters                 = 100
        self.T_max                          = 2
    
        self.update_params(**kwargs)




class Config3D(ConfigBase):
    def __init__(
        self, 
        name,
        data_dir=None,
        patch_size = [32, 96, 96],
        load_epoch=None,
        **kwargs
    ):
        super().__init__()

        self.delay_hours                    = 0.0

        self.name                           = name
        self.random_seed                    = 42


        self.log_dir                        =  './logs/'
        self.checkpoints_dir                =  "./checkpoints"
        self. result_dir                     =  "./results"

        # ========================= dataset ==================================

        self.data_dir                       = data_dir
        self.val_size                       =  0.15
        self.n_rays                         = 96
        self.foreground_prob                = 0.9
        self.n_classes                      = None # non None value (multiclass) not supported yet
        self.patch_size                     = patch_size
        self.cache_sample_ind               = True
        self.cache_data                     = True

        self.batch_size                     = 4
        self.num_workers                    = 0

        self.preprocess                     = "none"
        self.preprocess_val                 = "none"
        self.intensity_factor_range         = [0.6, 2.]
        self.intensity_bias_range           = [-0.2, 0.2]
        self.scale_limit                    = [1., 1.1]
        self.resize_to                      = [5, 286, 286]
        self.crop_size                      = [1, 256, 256]

        #======================================================================





        # ========================= Training ==================================

        self.use_gpu                        = True
        self.use_amp                        = True
        self.isTrain                        = True 
        self.evaluate                       = True #True
        #self.gpu_ids                       = [0]
        #self.continue_train                = False


        self.load_epoch                     = load_epoch
        self.n_epochs                       = 400
        self.n_steps_per_epoch              = 100


        self.lambda_prob                    = 1.
        self.lambda_dist                    = 0.1
        self.lambda_reg                     = 0.0001
        self.lambda_prob_class              = 1.

        self.save_epoch_freq                = 50
        self.start_saving_best_after_epoch  = 50 


        #======================================================================



        # ========================= Networks configurations ==================
        self.init_type                      = "normal"
        self.init_gain                      = 0.02

        self.backbone                       = "resnet"
        self.grid                           = "auto"
        self.anisotropy                     = "auto"

        self.n_channel_in                   = 1
        self.kernel_size                    = [3, 3, 3]
        self.resnet_n_blocks                = 3
        self.resnet_n_downs                 = None  # WILL BE SET to 'grid' in the code
        self.n_filter_of_conv_after_resnet  = 128
        self.resnet_n_filter_base           = 32
        self.resnet_n_conv_per_block        = 3
        self.use_batch_norm                 = False

        #======================================================================


        # ========================= Optimizers ================================
        self.lr                             = 0.0002
        self.beta1                          =  0.9
        self.beta2                          =  0.999

        self.lr_policy                      = "plateau"
        self.lr_plateau_factor              = 0.5
        self.lr_plateau_threshold           = 0.001
        self.lr_plateau_patience            = 40
        self.min_lr                         = 1e-6

        self.lr_linear_n_epochs             = 100
        self.lr_decay_iters                 = 100
        self.T_max                          = 2
    
        self.update_params(**kwargs)