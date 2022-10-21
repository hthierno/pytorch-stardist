import numpy as np

from stardist_tools.nms import non_maximum_suppression, non_maximum_suppression_sparse
from stardist_tools.geometry import polygons_to_label
from stardist_tools.matching import relabel_sequential


from .utils import with_no_grad
from .stardist_base import StarDistBase




class StarDist2D(StarDistBase):
    def __init__(self, opt):
        super().__init__(opt)

    @with_no_grad
    def predict_big(self, image, patch_size=(128, 128), context=(64, 64)):
        raise NotImplementedError("Per patch inference for 2D images not supported yet.")

    
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
        

        # Sparse prediction
        if points is not None:
            points, prob, dist, inds = non_maximum_suppression_sparse(
                dist, prob, points, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                prob_class = prob_class[inds]
        
        # Dense prediction
        else:
            points, prob, dist = non_maximum_suppression(
                dist, prob, grid=self.opt.grid, prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]
        


        labels = None

        if return_label_image:
            verbose = nms_kwargs.get('verbose',False)
            verbose and print("render polygons...")

            labels = polygons_to_label(
                dist, points, prob=prob, shape=img_shape, scale_dist=(1,1)
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
        

        res_dict = dict(dist=dist, points=points, prob=prob)#, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))
        
        return labels, res_dict