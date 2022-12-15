import itertools
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch
from torch.nn import init


######################################################################
#                    BaseNetwork and helper
######################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

class BaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def print_network(self):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))
    

    def init_net(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        Return an initialized network.
        """
        if len(gpu_ids)==1:
            assert(torch.cuda.is_available()), "cuda is not available."
            self.to(gpu_ids[0])

        elif len(gpu_ids) > 1:
            assert(torch.cuda.is_available()), "cuda is not available."
            self.to(gpu_ids[0])
            self = torch.nn.DataParallel(self, gpu_ids)  # multi-GPUs
        self.init_weights(init_type, init_gain=init_gain)


    def init_weights(self, init_type='normal', init_gain=0.02):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        self.apply(init_func)  # apply the initialization function <init_func>
        #return net
    
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, init_gain)



def define_stardist_net(opt):
    if opt.backbone != "resnet":
        raise NotImplementedError(f"<{opt.backbone}> is not supported. Backbone supported: <resnet>")
    
    hparams = {
        "input_nc": opt.n_channel_in,
        "n_rays": opt.n_rays,
        "n_classes": opt.n_classes,
        "n_filter_of_conv_after_resnet": opt.n_filter_of_conv_after_resnet,
        "n_blocks": opt.resnet_n_blocks,
        "n_downs": opt.resnet_n_downs,
        "n_filter_base": opt.resnet_n_filter_base,
        "n_conv_per_block": opt.resnet_n_conv_per_block,
        "kernel_size": opt.kernel_size,
        "batch_norm": opt.use_batch_norm,
        "activation": nn.ReLU(),
        "last_conv_bias_if_batch_norm": False
    }

    net = StarDistResnet( **hparams )

    gpu_ids = [0] if opt.use_gpu else []
    net.init_net( init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=gpu_ids )
    net.print_network()
    #net.hparams = hparams

    return net
        


######################################################################
#                              ResNet
######################################################################

class StarDistResnet(BaseNetwork):
    def __init__(
        self,
        input_nc,
        n_rays=96,
        n_classes=None,
        n_filter_of_conv_after_resnet=None,

        n_blocks=2,
        n_downs = (2, 2, 2),
        n_filter_base=32,
        n_conv_per_block=2,
        kernel_size=(3,3,3),
        batch_norm = False,
        activation = nn.ReLU(),
        last_conv_bias_if_batch_norm=False
    ):
        super(StarDistResnet, self).__init__()

        self.resnet = Resnet(
            input_nc=input_nc,
            output_nc=None,
            n_blocks=n_blocks,
            n_downs = n_downs,
            n_filter_base=n_filter_base,
            n_conv_per_block=n_conv_per_block,
            kernel_size=kernel_size,
            batch_norm=batch_norm,
            activation=activation,
            last_conv_bias_if_batch_norm=last_conv_bias_if_batch_norm
        )

        self.n_classes = n_classes
        n_dim = len(kernel_size)
        self.n_dim = n_dim
        conv_layer = nn.Conv2d if n_dim==2 else nn.Conv3d
        

        self.head_in_nc = self.resnet.output_nc
        conv_after_resnet = self.build_conv_after_resnet(n_filter_of_conv_after_resnet, conv_layer, kernel_size)

        self.rays_head = conv_layer( self.head_in_nc, n_rays, kernel_size=1, padding=0 )
        self.prob_head = conv_layer( self.head_in_nc, 1, kernel_size=1, padding=0 )
        self.prob_rays_conv_after_resnet = deepcopy( conv_after_resnet )
        
        if n_classes is not None:
            self.prob_class_head = [ conv_layer( self.head_in_nc, n_classes+1, kernel_size=1, padding=0 ) ]
            if not isinstance(conv_after_resnet, nn.Identity):
                self.prob_class_head = [ deepcopy(conv_after_resnet) ] + self.prob_class_head
        
        if n_classes is not None:
            self.prob_class_head = nn.Sequential(*self.prob_class_head)

    
    def build_conv_after_resnet(self, n_filter_of_conv_after_resnet, conv_layer, kernel_size):
        if (n_filter_of_conv_after_resnet is None) or n_filter_of_conv_after_resnet <= 0:
            return nn.Identity()
        
        self.head_in_nc = n_filter_of_conv_after_resnet

        n_dim = len(kernel_size)
        kernel_size = np.array(kernel_size)
        pad_size = list( zip( kernel_size[::-1] // 2, (kernel_size-1)[::-1]//2 ) )
        pad_size = itertools.chain( *pad_size )
        padding = nn.ConstantPad2d(pad_size, value=0.) if n_dim==2 else nn.ConstantPad3d(pad_size, value=0.)

        return nn.Sequential(*[
            padding,
            conv_layer(self.resnet.output_nc, n_filter_of_conv_after_resnet, kernel_size=kernel_size)
        ])


    def forward(self, x):
        out_resnet = self.resnet(x)

        in_prob_rays = self.prob_rays_conv_after_resnet(out_resnet)
        prob = self.prob_head(in_prob_rays)
        rays = self.rays_head(in_prob_rays)

        if self.n_classes is None:
            class_prob = None
        else:        
            class_prob = self.prob_class_head(out_resnet)
        
        return rays, prob, class_prob
    
    def predict(self, x):
        rays, prob, class_prob = self.forward(x)
        prob = torch.sigmoid(prob)
        if class_prob is not None:
            class_prob = torch.nn.functional.softmax(class_prob, dim=-(1+self.n_dim) )
        return rays, prob, class_prob

class Resnet(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc=None,
        n_blocks=2,
        n_downs = (2, 2, 2),
        n_filter_base=32,
        n_conv_per_block=2,
        kernel_size=(3,3,3),
        batch_norm = False,
        activation = nn.ReLU(),
        last_conv_bias_if_batch_norm=False
    ):

        super(Resnet, self).__init__()

        assert (np.log2(n_downs) <= n_blocks).all(), f" n_downs > n_blocks. The number of resnet blocks (n_blocks={n_blocks}) does not allow to perform n_downs={n_downs} downsampling. Set n_downs<=n_blocks."
        
        n_dim = len(kernel_size)
        assert n_dim in (2, 3), f'Resnet only 2d or 3d (kernel_size={kernel_size})'

        conv_layer = nn.Conv2d if n_dim==2 else nn.Conv3d

        n_filter = n_filter_base

        model = [
            conv_layer( input_nc, n_filter, kernel_size=7, padding=3 ),
            conv_layer( n_filter, n_filter, kernel_size=3, padding=1 )
        ]

        pooled = np.array([1]*n_dim)

        n_filter_next = n_filter
        for _ in range( n_blocks ):
            stride = 1 + ( np.asarray( n_downs ) > pooled )
            pooled *= 2
            if any( s>1 for s in stride ):
                n_filter_next *= 2
            model += [ 
                ResnetBlock(
                    n_filter, n_filter_next,
                    kernel_size=kernel_size,
                    stride=stride,
                    n_conv_per_block=n_conv_per_block,
                    batch_norm=batch_norm,
                    activation=activation,
                    last_conv_bias_if_batch_norm=last_conv_bias_if_batch_norm
                )
             ]

            n_filter = n_filter_next
        
        if output_nc is None:
            self.output_nc = n_filter
        else:
            self.output_nc = output_nc
            model += [
                conv_layer( n_filter, output_nc, kernel_size=1, padding=0 )
            ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    
    def __init__(
        self,
        n_filter_in,
        n_filter_out,
        kernel_size=(3,3,3),
        stride=(1,1,1),
        n_conv_per_block=2,
        batch_norm = False,
        activation=nn.ReLU(),
        last_conv_bias_if_batch_norm=False
    ):
        super(ResnetBlock, self).__init__()
        assert n_conv_per_block >= 2, 'required: n_conv_per_block >= 2'
        assert len(stride) == len(kernel_size), "kernel and stride sizes must match."
        n_dim = len(kernel_size)
        assert n_dim in (2, 3), 'resnet_block only 2d or 3d.'
        
        self.activation = activation
        self.conv_block, self.residual_block = self.build_conv_and_residual_block(
            n_filter_in,
            n_filter_out,
            kernel_size,
            stride,
            n_conv_per_block,
            batch_norm,
            activation,
            last_conv_bias_if_batch_norm
        )
        
    
    def build_conv_and_residual_block(self, n_filter_in, n_filter_out, kernel_size, stride, n_conv_per_block, batch_norm, activation, last_conv_bias_if_batch_norm):
        n_dim = len(kernel_size)
        kernel_size = np.array(kernel_size)
        pad_size = list( zip( kernel_size[::-1] // 2, (kernel_size-1)[::-1]//2 ) )
        pad_size = itertools.chain( *pad_size )
        
        conv_layer = nn.Conv2d if n_dim==2 else nn.Conv3d
        norm_layer = nn.BatchNorm2d if n_dim==2 else nn.BatchNorm3d
        
        padding = nn.ConstantPad2d(pad_size, value=0.) if n_dim==2 else nn.ConstantPad3d(pad_size, value=0.)
        
        conv_block = []
        
        # First conv
        conv_block += [ padding, conv_layer( n_filter_in, n_filter_out, kernel_size=kernel_size, stride=stride, bias=not batch_norm ) ]
        if batch_norm:
            conv_block += [ norm_layer ]
        conv_block += [ activation ]
            
        # middle conv
        for _ in range( n_conv_per_block -  2):
            conv_block += [ padding, conv_layer( n_filter_out, n_filter_out, kernel_size=kernel_size, bias=not batch_norm ) ]
            if batch_norm:
                conv_block += [ norm_layer ]
            conv_block += [ activation ]
        
        # last conv
        conv_block += [ padding, conv_layer( n_filter_out, n_filter_out, kernel_size=kernel_size, bias=not batch_norm ) ]
        if batch_norm:
            conv_block += [ norm_layer ]
        
        # residual block
        residual_block = None
        if any( s!=1 for s in stride ) or n_filter_in != n_filter_out:
            last_conv_bias = last_conv_bias_if_batch_norm if batch_norm else True
            residual_block = [ padding, conv_layer( n_filter_in, n_filter_out, kernel_size=kernel_size, stride=stride, bias=not batch_norm ) ]
        else:
            residual_block = [nn.Identity()]
            
        return nn.Sequential(*conv_block), nn.Sequential(*residual_block)
        
        
            
    def forward(self, x):
        out = self.conv_block(x) + self.residual_block(x)
        return out



######################################################################
#                   DIST LOSS
######################################################################

class DistLoss(nn.Module):
    def __init__(self, lambda_reg=0., norm_by_mask=True):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.criterion = nn.L1Loss(reduction="none")
        self.norm_by_mask = norm_by_mask
    
    def forward(self, input, target, mask=torch.tensor(1.), dim=1, eps=1e-9):
        actual_loss = mask * self.criterion( input, target )
        norm_mask = mask.mean() + eps if self.norm_by_mask else 1
        if self.lambda_reg > 0:
            reg_loss = (1 - mask) * torch.abs(input)

            loss = actual_loss.mean(dim=dim) / norm_mask + self.lambda_reg * reg_loss.mean(dim=dim)
        
        else:
            loss = actual_loss.mean(dim=dim) / norm_mask
        return loss.mean()