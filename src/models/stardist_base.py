import sys
import warnings

from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np

import torch
import torch.nn as nn

from stardist_tools.nms import _ind_prob_thresh
from stardist_tools.utils import _is_power_of_2, optimize_threshold
from scipy.ndimage import zoom


from .networks import define_stardist_net, DistLoss
from .logger import Logger
from .utils import with_no_grad, get_scheduler, update_lr, makedirs, _make_grid_divisible, load_json, save_json, makedirs

from stardist_tools.rays3d import Rays_GoldenSpiral, rays_from_json

from .config import Config2D, Config3D

        

class StarDistBase(nn.Module):
    
    def __init__(self, opt, rays=None):
        super().__init__()
        
        self.thresholds = dict (
            prob = 0.5,
            nms  = 0.4,
        )

        
        

        # if opt.rays_json['kwargs']["anisotropy"] is None:
        #     warnings.warn("Anisotropy was not set. Assuming isotropy; this may reduce performances. if it is not the case")

    
        self.opt = opt
        self.opt.n_dim = len(self.opt.kernel_size)
        self.isTrain = opt.isTrain
        self.use_amp = opt.use_amp if hasattr(opt, "use_amp") else False
        
        if self.use_amp and not opt.use_gpu:
            warnings.warn("GPU is not used (use_gpu=False), so `use_amp` is set to False")
            self.use_amp = False

        self.device = torch.device(f"cuda:0") if opt.use_gpu else torch.device("cpu")
        self.logger=Logger()

        # Define and load networks
        if hasattr(opt, "load_epoch") and opt.load_epoch not in (None, ""):
            self.opt.epoch_count = opt.load_epoch
            name = None
            if opt.load_epoch=="best":
                name="best"
            
            load_path=None
            if hasattr(self.opt, "load_path"):
                load_path = opt.load_path
            self.load_state(name=name, load_path=load_path)

            self.opt.n_dim = len(self.opt.kernel_size)

        else:
            if self.opt.n_dim==3:
                if rays is None:
                    if hasattr(opt, "rays_json"):
                        rays = rays_from_json( opt.rays_json )
                    elif hasattr(opt, 'n_rays'):
                        rays = Rays_GoldenSpiral( opt.n_rays, anisotropy=(opt.anisotropy if opt.anisotropy!="auto" else None) )
                    else:
                        rays = Rays_GoldenSpiral( 96, anisotropy=opt.anisotropy )
                
                opt.rays_json = rays.to_json()
                
                if opt.rays_json['kwargs']["anisotropy"] is None:
                    warnings.warn("Anisotropy was not set. Assuming isotropy; this may reduce performances. if it is not the case")
            
            self.opt.epoch_count = 0
            self.net = define_stardist_net(opt)

            if self.isTrain:
                self.set_optimizers()
                self.set_criterions()

    
    def set_optimizers(self):
        opt = self.opt
        self.optimizer = torch.optim.Adam( self.net.parameters() , lr=opt.lr, betas=(opt.beta1, opt.beta2) )
        self.lr_scheduler = get_scheduler(self.optimizer, opt, init_lr=opt.lr)
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    
    def set_criterions(self):
        opt = self.opt
        self.criterion_object = torch.nn.BCEWithLogitsLoss()
        self.criterion_class = torch.nn.CrossEntropyLoss()
        self.criterion_dist = DistLoss(lambda_reg = opt.lambda_reg)


    @with_no_grad
    def evaluate(self, batch, epoch=None):
        opt = self.opt
        if epoch is None:
            epoch = self.opt.epoch_count
        
        device = self.device

        image = batch['image'].to(device)
        dist = batch['dist'].to(device)
        prob = batch['prob'].to(device)

        prob_class = None
        if "prob_class" in batch:
            prob_class = batch["prob_class"].to(device)
            assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        else:
            assert opt.n_classes is None, f"'prob_class' is None in batch but opt.n_classes = {opt.n_classes} != None"

        batch_size = image.shape[0]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_dist, pred_prob, pred_prob_class = self.net(image)

            loss_dist = self.criterion_dist(pred_dist, dist, mask=prob)
            loss_prob = self.criterion_object(pred_prob, prob)
            loss_prob_class = torch.tensor(0.)
            if prob_class is not None:
                loss_prob_class = self.criterion_class(pred_prob_class, prob_class)
            
            loss = loss_prob * opt.lambda_prob + loss_dist * opt.lambda_dist + loss_prob_class * opt.lambda_prob_class

            
        
        self.logger.log("Val_loss", loss.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_prob", loss_prob.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_dist", loss_dist.item(), epoch=epoch, batch_size=batch_size)
        # if prob_class is not None:
        #     self.logger.log("Val_loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)


        return loss.item(), loss_dist.item(), loss_prob.item(), loss_prob_class.item()

    
    def optimize_parameters(self, batch, epoch=None):

        self.net.train()

        if epoch is None:
            epoch = self.opt.epoch_count
        
        opt = self.opt
        device= self.device

        image = batch['image'].to(device)
        dist = batch['dist'].to(device)
        prob = batch['prob'].to(device)

        prob_class = None
        if "prob_class" in batch:
            prob_class = batch["prob_class"].to(device)
            assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        else:
            assert opt.n_classes is None, f"'prob_class' is None in batch but opt.n_classes = {opt.n_classes} != None"

        #if prob_class is not None:
        #    assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        #    prob_class = prob_class.to(device)
        #else:
        #    assert opt.n_classes is None, f"'prob_class' i None in batch but opt.n_classes = {opt.n_classes} != None"
        
        batch_size = image.shape[0]
        

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_dist, pred_prob, pred_prob_class = self.net(image)

            loss_dist = self.criterion_dist(pred_dist, dist, mask=prob)
            loss_prob = self.criterion_object(pred_prob, prob)
            loss_prob_class = torch.tensor(0.)
            if prob_class is not None:
                loss_prob_class = self.criterion_class(pred_prob_class, prob_class)
            
            loss = loss_prob * opt.lambda_prob + loss_dist * opt.lambda_dist + loss_prob_class * opt.lambda_prob_class
        

        # Mixed Precision ======================
        self.optimizer.zero_grad()
        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()


        self.logger.log("loss", loss.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_prob", loss_prob.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_dist", loss_dist.item(), epoch=epoch, batch_size=batch_size)
        #if prob_class is not None:
            #self.logger.log("loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)
        

        
        return loss.item(), loss_dist.item(), loss_prob.item(), loss_prob_class.item()




    @with_no_grad
    def predict(self, image, patch_size=None, context=None):
        """
        parameters
        ----------
        image : np.ndarray
            image or volume. shape = (channel, height, width) if self.n_dim==2 and = (channel, depth, height, width) if self.ndim==3
        """

        if patch_size is not None and context is not None:
            return self.predict_big(image, patch_size=patch_size, context=context)
        else:
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            pred_dist, pred_prob, pred_prob_class = self.net.predict(image)

        pred_dist = np.moveaxis( pred_dist.cpu().numpy(), 1, -1 )
        pred_prob = np.moveaxis( pred_prob.cpu().numpy(), 1, -1 )
        if pred_prob_class is not None:
            pred_prob_class = np.moveaxis( pred_prob_class.cpu().numpy(), 1, -1 )


        return pred_dist[0], pred_prob[0], ( None if pred_prob_class is None else pred_prob_class[0] )
    
    @with_no_grad
    def predict_sparse(self, image, b=2, prob_thresh=None, patch_size=None, context=None):
        if prob_thresh is None:
            prob_thresh = self.thresholds["prob"]

        if patch_size is not None and context is not None:
            print(" === Per patch inference")
            #dist, prob, prob_class = self.predict_big(image, patch_size=patch_size, context=context)
        # else:
        #     dist, prob, prob_class = self.predict(image)
        dist, prob, prob_class = self.predict(image, patch_size=patch_size, context=context)
        
        # ...
        prob = prob[..., 0]
        #dist = np.moveaxis(dist, 0, -1)
        dist = np.maximum( 1e-3, dist )
        #
        inds = _ind_prob_thresh(prob, prob_thresh, b=b)
        
        prob = prob[inds].copy()
        dist = dist[inds].copy()
        points = np.stack( np.where(inds), axis=1 )
        points = points * np.array( self.opt.grid ).reshape( 1, len(self.opt.grid) ) 

        if self._is_multiclass():
            assert prob_class is not None, f"prediction 'prob_class' is None but self.is_multiclass()==True"
            #prob_class = np.moveaxis(prob_class, 0, -1)
            prob_class = prob_class[inds].copy()
        
        return dist, prob, prob_class, points

    def _prepare_patchsize_context(self, patch_size, context):
        if context is None:
            if not hasattr(self, 'receptive_field') or self.receptive_field is None:
                self.receptive_field = [ max(rf) for rf in self._compute_receptive_field() ]
            
            context = self.receptive_field
        grid = self.opt.resnet_n_downs
        patch_size = _make_grid_divisible(grid, patch_size, name="patch_size")
        context = _make_grid_divisible(grid, context, name="context")

        print(context, patch_size, grid)
        
        return patch_size, context


    @with_no_grad
    def predict_instance(
        self,
        image,
        prob_thresh=None,
        nms_thresh=None,
        sparse = True,

        patch_size = None,
        context=None,

        return_label_image=True,
        return_predict=None,
        overlap_label=None,
        predict_kwargs=None, nms_kwargs=None
    ):
        """
        patch_size: tuple of size nb dim of image
            patch size to use for per patch prediction (to avoid OOM)
            default: None -> Perform one pass on the whole image

        context: tuple of size nb dim of image
            size of context to use around each patch during per patch_prediction
            default: None -> Use the model receptive field
        """

        self.net.eval()

        if patch_size is not None:
            patch_size, context = self._prepare_patchsize_context(patch_size, context)

        if predict_kwargs is None:
            predict_kwargs = dict()
        if nms_kwargs is None:
            nms_kwargs = dict()
        

        if sparse:
            dist, prob, prob_class, points = self.predict_sparse(image, patch_size=patch_size, context=context, **predict_kwargs)
            
        else:
            dist, prob, prob_class = self.predict(image, patch_size=patch_size, context=context, **predict_kwargs)
            prob = prob[..., 0] # removing the channel dimension
            points = None
        
        res = dist, prob, prob_class, points
        
        #print(dist.shape, prob.shape, prob_class.shape if prob_class is not None else None, points.shape if points is not None else None)
        

        shape = image.shape[-self.opt.n_dim:]
        
        res_instances = self._instances_from_prediction(
            shape,
            prob,
            dist,
            points=points,
            prob_class=prob_class,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            return_label_image=return_label_image,
            overlap_label=overlap_label,
            **nms_kwargs
        )

        if return_predict:
            return res_instances, tuple( res[:-1] )
        
        else:
            return res_instances    


    
    def _is_multiclass(self):
        return self.opt.n_classes is not None
        
    
    
    def _compute_receptive_field(self, img_size=None):
        # TODO: good enough?
        if img_size is None:
            img_size = tuple(g*(128 if self.opt.n_dim==2 else 64) for g in self.opt.grid)
        if np.isscalar(img_size):
            img_size = (img_size,) * self.opt.n_dim
        img_size = tuple(img_size)
        # print(img_size)

        assert all(_is_power_of_2(s) for s in img_size)

        mid = tuple(s//2 for s in img_size)
        x = np.zeros( (1, self.opt.n_channel_in) + img_size, dtype=np.float32)
        z = np.zeros_like(x)
        x[ (0,slice(None)) + mid ] = 1
        
        with torch.no_grad():
            y  = self.net.forward( torch.from_numpy(x).to(self.device) )[0][0,0].cpu().numpy()
            y0 = self.net.forward( torch.from_numpy(z).to(self.device) )[0][0,0].cpu().numpy()

        grid = tuple( ( np.array(x.shape[2:]) / np.array(y.shape) ).astype(int) )
        assert grid == self.opt.grid, ( grid, self.opt.grid )
        y  = zoom( y, grid,order=0 )
        y0 = zoom( y0, grid,order=0 )
        ind = np.where( np.abs(y-y0)>0 )
        return [ ( m-np.min(i), np.max(i)-m ) for (m,i) in zip(mid,ind) ]
    

    def update_lr(self, epoch=None, metric=None, metric_name=""):
        if epoch is None:
            epoch = self.opt.epoch_count

        lr = update_lr(self.optimizer, self.lr_scheduler, self.opt, metric=metric)

        self.logger.log("lr", lr, epoch=epoch, batch_size=1)
        if metric is not None:
            self.logger.log(f"{metric_name}_metric", float(metric), epoch=epoch, batch_size=1)
    

    def save_state(self, name=None):
        save_path = Path( self.opt.checkpoints_dir ) / f"{self.opt.name}"
        makedirs(save_path)

        epoch = self.opt.epoch_count
        
        state = {
            "opt": vars(self.opt),
            "epoch": epoch,

            "model_state_dict": self.net.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "amp_scaler_state_dict": self.amp_scaler.state_dict(),
        }


        if name is None:
            name = f"epoch{epoch}_ckpt"
        torch.save( state, save_path / f"{name}.pth" )


        print(f"Networks saved at <{save_path}>")

        log_dir = Path(self.opt.log_dir) / f"{self.opt.name}"
        self.logger.to_pickle( path= log_dir / "metrics_logs.pkl" )
        self.logger.to_csv( path= log_dir / "metrics_logs.csv" )
        
        save_json(log_dir /'last_configuration.json', self.thresholds)

        # with open(log_dir /'last_configuration.yml', 'w') as f:
        #     yaml.dump( vars(self.opt), stream=f )


        print(f"Logger saved at <{log_dir}>")

        self.net.to(self.device)




    def load_state(self, name=None, load_path=None):
        opt = self.opt
        
        if load_path is None:
            if name is None:
                name = f"epoch{opt.epoch_count}_ckpt"

            load_dir = Path(opt.checkpoints_dir) / f"{opt.name}"        
            load_path =  load_dir / f"{name}.pth"
        
        print('Load path:', load_path, self.device)
        state = torch.load(load_path, map_location=str(self.device) )

        loaded_opt = state["opt"]
        config_class = Config3D if loaded_opt['n_dim']==3 else Config2D
        
        loaded_opt = config_class(allow_new_params=True, **loaded_opt)
        loaded_opt.epoch_count = state['epoch']
        loaded_opt.name = opt.name
        loaded_opt.use_gpu = opt.use_gpu
        loaded_opt.use_amp = opt.use_amp
        loaded_opt.checkpoints_dir = opt.checkpoints_dir
        loaded_opt.log_dir = opt.log_dir
        

        
        if opt.n_epochs > loaded_opt.n_epochs:
            loaded_opt.n_epochs = opt.n_epochs
        self.opt = loaded_opt

        if self.opt.n_dim==3:
            if self.opt.rays_json['kwargs']["anisotropy"] is None:
                warnings.warn("Anisotropy is not in the checkpoint. Assuming isotropy; This may reduce performances if it is not the case.")

        
        ################################

        ### Loading thresholds
        checkpoint_dir = Path( self.opt.checkpoints_dir ) / f"{self.opt.name}"
        thresholds_path = checkpoint_dir / "thresholds.json"
        if thresholds_path.exists():
            print("Loading threholds ...")
            self.thresholds = load_json(thresholds_path)
        else:
            warnings.warn(f"Didn't find thresholds in checkpoint at <{thresholds_path}>. Using Default Thresholds: {self.thresholds}")
        
        print("Instanciating network")
        self.net = define_stardist_net( loaded_opt )
        print( self.net.load_state_dict( state['model_state_dict'] ) )

        
        if opt.isTrain:
            
            self.set_optimizers()
            self.set_criterions()

            if not ( hasattr(opt, "reset_optimizers") and opt.reset_optimizers ):

                

                self.optimizer.load_state_dict( state['optimizer_state_dict'] )
                self.lr_scheduler.load_state_dict( state['scheduler_state_dict'] )


                if "amp_scaler_state_dict" in state:
                    self.amp_scaler.load_state_dict( state['amp_scaler_state_dict'] )
                    print("Optimizers, schedulers and amp_scaler loaded.")

                else:
                    print("*** amp_scaler not in checkpoint. Initialize a new amp_scaler !!!")
                    print("Optimizers and schedulers loaded.")


            else:
                print(f"opt.reset_optimizers={opt.reset_optimizers}. Optimizers and Schedulers don't loaded.")


            self.logger.load_pickle(Path(opt.log_dir) / f"{opt.name}/metrics_logs.pkl", epoch=self.opt.epoch_count)
            print("Logger loaded.")

        print(f"Loading model from <{load_path}>.\n")



    
    
    def optimize_thresholds(
        self,
        X_val,
        Y_val,
        nms_threshs=[0.3,0.4,0.5],
        iou_threshs=[0.3,0.5,0.7],
        predict_kwargs=None,
        optimize_kwargs=None,
        #save_to_json=True
    ):
        # Modified from https://github.com/stardist/stardist/blob/master/stardist/models/base.py
        """Optimize two thresholds (probability, NMS overlap) necessary for predicting object instances.
        Note that the default thresholds yield good results in many cases, but optimizing
        the thresholds for a particular dataset can further improve performance.
        The optimized thresholds are automatically used for all further predictions
        and also written to the model directory.
        See ``utils.optimize_threshold`` for details and possible choices for ``optimize_kwargs``.
        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        Y_val : list of ndarray
            (Validation) label images to use for threshold tuning.
        nms_threshs : list of float
            List of overlap thresholds to be considered for NMS.
            For each value in this list, optimization is run to find a corresponding prob_thresh value.
        iou_threshs : list of float
            List of intersection over union (IOU) thresholds for which
            the (average) matching performance is considered to tune the thresholds.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.
        """
        self.net.eval()

        if predict_kwargs is None:
            predict_kwargs = dict()
        else:
            if "patch_size" in predict_kwargs and predict_kwargs["patch_size"] is not None:
                predict_kwargs = deepcopy(predict_kwargs)
                patch_size = predict_kwargs["patch_size"]
                context = predict_kwargs.get("context", None)
                patch_size, context = self._prepare_patchsize_context(patch_size, context)
                predict_kwargs["patch_size"] = patch_size
                predict_kwargs["context"] = context

        if optimize_kwargs is None:
            optimize_kwargs = dict()
        



        # only take first two elements of predict in case multi class is activated
        #Yhat_val = [self.predict(x, **_predict_kwargs(x))[:2] for x in X_val]
        
        pred_prob_dist = []
        for x in X_val:
            dist, prob = self.predict(x, **predict_kwargs)[:2]
            dist = np.maximum( 1e-3, dist )
            
            prob = prob[..., 0] # removing the channel dimension
            pred_prob_dist.append( [ prob, dist ] )
        

        opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None
        
        for _opt_nms_thresh in nms_threshs:
            
            _opt_prob_thresh, _opt_measure = optimize_threshold(
                Y_val, Yhat=pred_prob_dist,
                model=self,
                nms_thresh=_opt_nms_thresh,
                iou_threshs=iou_threshs,
                **optimize_kwargs
            )
            
            if _opt_measure > opt_measure:
                opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh
        
        opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)

        self.thresholds = opt_threshs
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=self.thresholds["prob"], nms=self.thresholds["nms"]))
        
        # log_dir = Path(self.opt.log_dir) / f"{self.opt.name}"
        checkpoint_dir = Path( self.opt.checkpoints_dir ) / f"{self.opt.name}"
        dest_path = checkpoint_dir / "thresholds.json"
        makedirs(dest_path)
        print(f"Saving to <{dest_path}>")
        save_json(dest_path, self.thresholds)
        
        return opt_threshs
