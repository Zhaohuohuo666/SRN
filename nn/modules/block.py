# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.ops import DeformConv2d
from einops import rearrange

from timm.models.layers import create_act_layer, get_act_layer
from timm.models.layers import make_divisible
from timm.models.layers import ConvMlp
from timm.models.layers import LayerNorm2d
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock
from ultralytics.nn.BiFormer import BiLevelRoutingAttention, Attention, AttentionLePE

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        #self.m = nn.ModuleList(Focalnext(self.c, self.c) for _ in range(n))
        #self.m = nn.ModuleList(Bottleneck_GS(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


#############################c2f_SGC###########################################
class C2f_BGNgc(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_GNgc(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
#################################################################################
    

#############################c2f_BGNCA###########################################
class C2f_BGNCA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_GNCA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        #self.m = nn.ModuleList(Focalnext(self.c, self.c) for _ in range(n))
        #self.m = nn.ModuleList(Bottleneck_GS(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
#################################################################################



    

class C2f_focalnext(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        #self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m = nn.ModuleList(FocalNext(self.c, self.c) for _ in range(n))
        #self.m = nn.ModuleList(Bottleneck_GS(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    


########################################################################
class C2f_CGBLOCK(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ContextGuidedBlock(self.c, self.c, dilation_rate=2, reduction=16, add=True) for _ in range(n))
        
        #self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
########################################################################






class C2f_DWRBLOCK(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(DWR_Block(self.c, self.c) for _ in range(n))
        
        #self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


#####################DWR_Block#########################################
class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out
    
class Dilated_DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, d=1):
        super(Dilated_DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, dilation=d)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out


class DWR_Block(nn.Module):
    def __init__(self, nIn, nOut):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        self.n= int(nIn/2)
        self.conv3x3 = ConvBNPReLU(nIn, int(3*self.n), 3, 1)  #1x1 Conv is employed to reduce the computation
        self.D1 = DConv(self.n, int(2*self.n), 3, 1)
        self.D2 = Dilated_DConv(self.n, self.n, 3, padding=3, d=3)
        self.D3 = Dilated_DConv(self.n, self.n, 3, padding=5, d=5)
        self.conv1x1 = nn.Conv2d(int(4*self.n), nOut, 1)
        self.bn = nn.BatchNorm2d(int(4*self.n), eps=1e-03)
        # self.F_loc = ChannelWiseConv(self.n, self.n, 3, 1) # local feature
        # self.F_sur = ChannelWiseDilatedConv(self.n, self.n, 3, 1, dilation_rate) # surrounding context

    def forward(self, input):
        input1 = self.conv3x3(input)#此时通道数变为输入通道数的3/2倍
        y = list(input1.split((self.n, self.n, self.n), 1))
        y1 = self.D1(y[0])
        y2 = self.D2(y[1])
        y3 = self.D3(y[2])
        
        joi_feat = torch.cat([y1, y2, y3], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.conv1x1(joi_feat)
        output = joi_feat+input
        return output
######################################################################################



#################################上采样算子carafe###################################
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X
##################################################################################



###############################SCconv#####################################
class GroupBatchnorm2ds(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2ds,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.beta       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRUs(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5 
                 ):
        super().__init__()
        
        self.gn             = GroupBatchnorm2ds( oup_channels, group_num = group_num )
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.gamma/sum(self.gn.gamma)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = reweigts>=self.gate_treshold
        noninfo_mask= reweigts<self.gate_treshold
        x_1         = info_mask * x
        x_2         = noninfo_mask * x
        x           = self.reconstruct(x_1,x_2)
        return x
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   int(op_channel-up_channel)
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 16,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRUs = SRUs( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRUs(x)
        x = self.CRU(x)
        return x
##########################################################################







#################################RFB_block##################################
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        out = out*self.scale + x
        out = self.relu(out)

        return out
############################################################################






##################################CGBLOCK##################################
class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class ConvBNGeLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.GELU()

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                #nn.Conv2d(channel, channel//reduction, 1),
                #LayerNorm2d(channel//reduction),
                nn.ReLU(inplace=True),
                #nn.GELU(),
                #nn.Dropout(),
                nn.Linear(channel // reduction, channel),
                #nn.Conv2d(channel // reduction, channel, 1)
                nn.Sigmoid()
                #nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class BNGeLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.GELU()

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output
class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=1, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


    







####################CA_Block###############################
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class CA(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(CA, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 4
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        #self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
        self.LR = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.LayerNorm([self.channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.in_channels, kernel_size=1),
        )

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        #constant_init(self.m.conv, 0)
        last_zero_init(self.LR)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        #y = self.m(y) * a
        y = self.LR(y) * a

        return y
######################################################################











################################CE_concat#################################
class Ce_Concat(nn.Module):
    def __init__(self, inc, channel):
        super(Ce_Concat, self).__init__()
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv0 = nn.Conv2d(inc[0], int(channel/2),kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(inc[1], int(channel/2),k=1, s=1, p=0)

    def forward(self, x):
        p1, p2 = x[0],x[1]
        y1 = self.conv0(p1)
        y2_1 = self.m1(p2)
        y2_2 = self.m2(p2)
        y2 = y2_1+y2_2
        y2 = self.conv1(y2)
        output = torch.concat([y1, y2], dim=1)
        return output
##########################################################################






class space_to_depth(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    


# ###########################RFAConv##########################################
class RFAConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel,bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())
       
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size))

    def forward(self,x):
        b,c = x.shape[0:2]
        weight =  self.get_weight(x)
        h,w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2) 
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        return self.conv(conv_data)


class RFSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    # act参数在yolov7-tiny上记得修改为nn.LeakyReLU(0.1)
    default_act = nn.SiLU
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.up_channel = 0.5*c2
        self.low_channel = 0.5*c2
        self.addcoords = AddCoords(with_r=False)
        self.squeeze0 = nn.Conv2d(4*int(c1), int(c2), kernel_size=1, stride=1, bias=False)
        #self.cv1 = nn.Conv2d(int(self.up_channel), int(self.up_channel/2), kernel_size=k, stride=s, padding=1) # 第一个卷积操作
        # self.GWC = nn.Conv2d(c_, c_, kernel_size=3, stride=1,
        #                      padding=3 // 2, groups=2)
        # self.PWC1 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)

        # self.PWC2 = nn.Conv2d(int(c_ / 2), int(c_ / 2), kernel_size=1,
        #                       bias=False)
        self.space_to_depth = space_to_depth(1)
        self.RFAC = RFAConv(int(self.up_channel+2), int(self.up_channel), 3, 1)
        #self.SPC = SPConv_3x3(int(self.up_channel), int(self.up_channel))
        #self.cv2 = nn.Conv2d(int(self.up_channel/2), int(self.up_channel/2), 5, 1, padding=2, groups = int(self.up_channel/2))
        # self.cv3 = Conv(int((c1)), int(c2), 1)

    def forward(self, x):
        x = self.space_to_depth(x)
        x = self.squeeze0(x)
        up, low = torch.split(x, [int(self.up_channel), int(self.low_channel)], dim=1) #将经过spd操作的特征图进行拆分操作
        ##############low特征图不进行操作，只针对up特征图进行操作###################################
        up = self.addcoords(up) #对up特征图添加坐标信息

        # x1 = self.cv1(up)# 第一次进行卷积操作，经过卷积操作,通道数变为一半，卷积核大小为3*3
        # x2 = x1
        # x2 = self.cv2(x2)
        # up = torch.cat([x1, x2], dim=1)
        up = self.RFAC(up)
        y = torch.cat([up, low], dim=1)


        b, n, h, w = y.size()
        b_n = b * n // 2
        y = y.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)
        # y = self.cv3(y)

        return y



class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class DilatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, d = 2, p = 1):
        super(DilatedDepthwiseSeparableConv, self).__init__()

        # 深度扩张卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding=p, groups=in_channels, dilation=d),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class FocalNext(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(FocalNext, self).__init__()
        self.DWConv = DepthwiseSeparableConv(in_planes, out_planes)
        #self.D_Dwconv1 = DilatedDepthwiseSeparableConv(in_planes, out_planes, d=1, p=1)
        self.D_Dwconv2 = DilatedDepthwiseSeparableConv(in_planes, out_planes, d=4, p=4)
        #self.D_Dwconv3 = DilatedDepthwiseSeparableConv(in_planes, out_planes, d=3, p=3)
        self.Conv1 = Conv(in_planes, int(in_planes/2))
        self.Conv2 = Conv(int(in_planes/2), out_planes)

    def forward(self, x):
        y1 = self.DWConv(x)
        y1 = x+y1
        #y2_1 = self.D_Dwconv1(y1)
        y2_2 = self.D_Dwconv2(y1)
        #y2_3 = self.D_Dwconv3(y1)
        y2 = y1+y2_2
        y2 = self.Conv1(y2)
        y2 = self.Conv2(y2)
        output = y2+x
        return output
    




################################Bottleneck_sc############################
class Bottleneck_sc(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, shortcut = True):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.scconv = ScConv(c1)
        #self.cruconv = cruconv(c1)
        #self.add = shortcut and c1 == c2
        self.ATT = GN_gc(c1)

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        #return self.ATT(x) + self.scconv(x)

        return self.ATT(x) + self.scconv(x)
        #return x + self.scconv(self.scconv(x))
#########################################################################