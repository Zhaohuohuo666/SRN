# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math
import pkg_resources as pkg
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F

from ultralytics.utils.tal import dist2bbox, make_anchors
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_
from ultralytics.nn.dyhead import DyHeadBlock
__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))
class Detect_LSCD(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()
        
    def init_weights(self):
        # self.la_conv1.weight.normal_(std=0.001)
        # self.la_conv2.weight.normal_(std=0.001)
        # self.la_conv2.bias.data.zero_()
        # self.reduction_conv.conv.weight.normal_(std=0.01)
        
        torch.nn.init.normal_(self.la_conv1.weight.data, mean=0, std=0.001)
        torch.nn.init.normal_(self.la_conv2.weight.data, mean=0, std=0.001)
        torch.nn.init.zeros_(self.la_conv2.bias.data)
        torch.nn.init.normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat
class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        #self.cv4 = nn.ModuleList(Conv(x, 128, 1) for x in ch)
        #self.cv5 = nn.ModuleList(Conv(128, x, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # self.dyhead = nn.Sequential(*[DyHeadBlock(ch[0]) for i in range(1)])

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # for i in range(self.nl):
        #     x[i] = (self.cv4[i](x[i]))
        # for dyhead_layer in self.dyhead:
        #     x = dyhead_layer(x)
        # for i in range(self.nl):
        #     x[i] = (self.cv5[i](x[i]))
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        
        
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    # dynamic = False  # force grid reconstruction
    # export = False  # export mode
    # shape = None
    # anchors = torch.empty(0)  # init
    # strides = torch.empty(0)  # init

    # def __init__(self, nc=80, hidc=256, ch=()):
    #     """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
    #     super().__init__()
    #     self.nc = nc  # number of classes
    #     self.nl = len(ch)  # number of detection layers
    #     self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    #     self.no = nc + self.reg_max * 4  # number of outputs per anchor
    #     self.stride = torch.zeros(self.nl)  # strides computed during build
    #     self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
    #     self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))
    #     self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
    #     self.cv3 = nn.Conv2d(hidc, self.nc, 1)
    #     self.scale = nn.ModuleList(Scale(1.0) for x in ch)
    #     self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     for i in range(self.nl):
    #         x[i] = self.conv[i](x[i])
    #         x[i] = self.share_conv(x[i])
    #         x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
    #     if self.training:  # Training path
    #         return x

    #     # Inference path
    #     shape = x[0].shape  # BCHW
    #     x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    #     if self.dynamic or self.shape != shape:
    #         self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    #         self.shape = shape

    #     if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
    #         box = x_cat[:, : self.reg_max * 4]
    #         cls = x_cat[:, self.reg_max * 4 :]
    #     else:
    #         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    #     dbox = self.decode_bboxes(box)

    #     if self.export and self.format in ("tflite", "edgetpu"):
    #         # Precompute normalization factor to increase numerical stability
    #         # See https://github.com/ultralytics/ultralytics/issues/7371
    #         img_h = shape[2]
    #         img_w = shape[3]
    #         img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
    #         norm = self.strides / (self.stride[0] * img_size)
    #         dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

    #     y = torch.cat((dbox, cls.sigmoid()), 1)
    #     return y if self.export else (y, x)

    # def bias_init(self):
    #     """Initialize Detect() biases, WARNING: requires stride availability."""
    #     m = self  # self.model[-1]  # Detect() module
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
    #     # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
    #     # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
    #     m.cv2.bias.data[:] = 1.0  # box
    #     m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    #Task Dynamic Align Detection Head
    """YOLOv8 Detect head for detection models."""

    # dynamic = False  # force grid reconstruction
    # export = False  # export mode
    # shape = None
    # anchors = torch.empty(0)  # init
    # strides = torch.empty(0)  # init

    # def __init__(self, nc=80, hidc=256, ch=()):
    #     """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
    #     super().__init__()
    #     self.nc = nc  # number of classes
    #     self.nl = len(ch)  # number of detection layers
    #     self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    #     self.no = nc + self.reg_max * 4  # number of outputs per anchor
    #     self.stride = torch.zeros(self.nl)  # strides computed during build
    #     self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
    #     self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
    #     self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
    #     self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
    #     self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
    #     self.offset_dim = 2 * 3 * 3
    #     self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
    #     self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
    #     self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
    #     self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
    #     self.scale = nn.ModuleList(Scale(1.0) for x in ch)
    #     self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     for i in range(self.nl):
    #         stack_res_list = [self.share_conv[0](x[i])]
    #         stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
    #         feat = torch.cat(stack_res_list, dim=1)
            
    #         # task decomposition
    #         avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
    #         cls_feat = self.cls_decomp(feat, avg_feat)
    #         reg_feat = self.reg_decomp(feat, avg_feat)
            
    #         # reg alignment
    #         offset_and_mask = self.spatial_conv_offset(feat)
    #         offset = offset_and_mask[:, :self.offset_dim, :, :]
    #         mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
    #         reg_feat = self.DyDCNV2(reg_feat, offset, mask)
            
    #         # cls alignment
    #         cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()
            
    #         x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)
    #     if self.training:  # Training path
    #         return x

    #     # Inference path
    #     shape = x[0].shape  # BCHW
    #     x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    #     if self.dynamic or self.shape != shape:
    #         self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    #         self.shape = shape

    #     if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
    #         box = x_cat[:, : self.reg_max * 4]
    #         cls = x_cat[:, self.reg_max * 4 :]
    #     else:
    #         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    #     dbox = self.decode_bboxes(box)

    #     if self.export and self.format in ("tflite", "edgetpu"):
    #         # Precompute normalization factor to increase numerical stability
    #         # See https://github.com/ultralytics/ultralytics/issues/7371
    #         img_h = shape[2]
    #         img_w = shape[3]
    #         img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
    #         norm = self.strides / (self.stride[0] * img_size)
    #         dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

    #     y = torch.cat((dbox, cls.sigmoid()), 1)
    #     return y if self.export else (y, x)

    # def bias_init(self):
    #     """Initialize Detect() biases, WARNING: requires stride availability."""
    #     m = self  # self.model[-1]  # Detect() module
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
    #     # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
    #     # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
    #     m.cv2.bias.data[:] = 1.0  # box
    #     m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    # def decode_bboxes(self, bboxes):
    #     """Decode bounding boxes."""
    #     return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

class CLLA(nn.Module):
    def __init__(self, range, c):
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
        # x2 = x2.squeeze(3).permute(0, 3, 1, 2).contiguous()
        return (out + x2) / 2
        # return out
class CLLABlock(nn.Module):
    def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0):
        super().__init__()
        self.range = range
        self.c_ = ch
        self.cout = out
        self.conv1 = nn.Conv2d(ch1, self.c_, 1)
        self.conv2 = nn.Conv2d(ch2, self.c_, 1)

        self.att = CLLA(range = range, c = self.c_)

        self.det = nn.Conv2d(self.c_, out, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        f = self.att(x1, x2)

        return self.det(f)

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result

class CLLADetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.det = CLLABlock(range = 2, ch = ch[0], ch1 = ch[0], ch2 = ch[1], out = self.no * self.na)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[2:])  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        p = []
        for i in range(self.nl):
            if i == 0:
                p.append(self.det(x[0], x[1]))
            else:
                p.append(self.m[i-1](x[i+1]))  # conv
            bs, _, ny, nx = p[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != p[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = p[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return p if self.training else (torch.cat(z, 1), p)
        # return x if self.training else (z[0], x) # (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid







class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
