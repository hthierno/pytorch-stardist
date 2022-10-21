import numpy as np


from stardist_tools.rays3d import rays_from_json
from stardist_tools.nms import non_maximum_suppression_3d, non_maximum_suppression_3d_sparse
from stardist_tools.geometry import polyhedron_to_label
from stardist_tools.matching import relabel_sequential

from .utils import Block3D, with_no_grad
from .stardist_base import StarDistBase




class StarDist3D(StarDistBase):
    def __init__(self, opt):
        super().__init__(opt)

        
    
    @with_no_grad
    def predict_big(self, image, patch_size=(32, 128, 128), context=(32, 64, 64)):
        """
        parameters
        ----------
        image : np.ndarray
            volume of shape = (channel, depth, height, width)
        """
        #print((image.shape, self.opt.resnet_n_downs))
        grid = self.opt.resnet_n_downs

        assert (image.ndim - 1) == len( grid ), (image.shape, grid)

        prob_block = Block3D( image, n_channel_target=1, grid=grid,
            patch_size=patch_size,
            context=context
        )

        dist_block = Block3D( image, n_channel_target=self.opt.n_rays, grid=grid,
            patch_size=patch_size,
            context=context
        )

        assert self.opt.n_classes is None, f"`predict_big` don't support yet multiclass predication"




        for (idx_prob, image_prob), (idx_dist, image_dist) in zip( prob_block, dist_block ):
            assert image_prob.shape == image_dist.shape, (image_prob.shape, image_dist.shape)
            assert idx_prob == idx_dist, (idx_prob, idx_dist)

            idx = idx_prob
            image_block = image_prob

            pred_dist, pred_prob, _ = self.predict(image_block)
            pred_dist = np.moveaxis( pred_dist, -1, 0)
            pred_prob = np.moveaxis( pred_prob, -1, 0)

            #print(pred_prob.shape, pred_dist.shape, "ok")

            prob_block.set_target_patch(idx, pred_prob)
            dist_block.set_target_patch(idx, pred_dist)
        
        final_pred_prob = prob_block.get_target_volume()
        final_pred_dist = dist_block.get_target_volume()

        final_pred_prob = np.moveaxis( final_pred_prob, 0, -1)
        final_pred_dist = np.moveaxis( final_pred_dist, 0, -1)

        return final_pred_dist, final_pred_prob, None
    
    def _instances_from_prediction(
        self,
        img_shape,
        prob,
        dist,
        points=None,
        prob_class=None,
        prob_thresh=None,
        nms_thresh=None,
        return_label_image=True,
        overlap_label=None,
        **nms_kwargs
    ):

        
        self.net.eval()

        if prob_thresh is None:
            prob_thresh = self.thresholds['prob']
        if nms_thresh is None:
            nms_thresh = self.thresholds['nms']
        
        rays = rays_from_json( self.opt.rays_json )

        #assert prob.ndim == 3, prob.shape
        #assert dist.ndim == 4 and dist.shape[-1] == len(rays), (dist.shape, len(rays) )
        #assert prob.shape == dist.shape[:3], (prob.shape, dist.shape)

        # Sparse prediction
        if points is not None:
            points, prob, dist, inds = non_maximum_suppression_3d_sparse(
                dist, prob, points, rays, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                prob_class = prob_class[inds]
        
        # Dense prediction
        else:
            points, prob, dist = non_maximum_suppression_3d(
                dist, prob, rays, grid=self.opt.grid, prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]
        


        labels = None

        if return_label_image:
            verbose = nms_kwargs.get('verbose',False)
            verbose and print("render polygons...")

            labels = polyhedron_to_label(
                dist, points, rays=rays, prob=prob, shape=img_shape, overlap_label=overlap_label, verbose=verbose
            )

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        

        res_dict = dict(dist=dist, points=points, prob=prob, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))
        
        return labels, res_dict