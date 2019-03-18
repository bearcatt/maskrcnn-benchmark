# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C


class _GlobalROIPool(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        ctx.output_size = _pair(output_size)
        ctx.input_shape = input.size()
        output = _C.global_roi_pool_forward(
            input, output_size[0], output_size[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        output_size = ctx.output_size
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.global_roi_pool_backward(
            grad_output,
            output_size[0],
            output_size[1],
            bs, ch, h, w,
        )
        return grad_input, None


global_roi_pool = _GlobalROIPool.apply


class Global_ROI_Pool(nn.Module):
    def __init__(self, output_size):
        super(Global_ROI_Pool, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return global_roi_pool(input, self.output_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ")"
        return tmpstr
