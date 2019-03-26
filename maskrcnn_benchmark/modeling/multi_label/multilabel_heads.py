import torch
import numpy as np

from torch import nn
from maskrcnn_benchmark.layers import Global_ROI_Pool

from maskrcnn_benchmark.modeling.backbone import resnet

class MultiLabelCls(nn.Module):
    """
    Module that adds multi-label classifier on top of the feature maps.
    """
    def __init__(self, config):
        super(MultiLabelCls, self).__init__()
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )
        self.pooler = nn.AdaptiveAvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1 # 80
        # 1024 for both c4 and FPN architecture
        num_inputs = config.MODEL.RESNETS.BACKBONE_OUT_CHANNELS * 2
        self.classifier = nn.Linear(num_inputs, self.num_classes)

        nn.init.normal_(self.classifier.weight, mean=0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

        self.softmax = nn.Softmax()
    
    def forward(self, x, labels_list):
        """
        Args:
            x: (N, C, H, W) float tensor
            labels_list: list of (?,) float tensor
        """
        labels_onehot_list = []
        for labels in labels_list:
            labels_onehot = labels.new(self.num_classes).zero_().float()
            labels_onehot.scatter_(0, labels - 1, 1.0)
            labels_onehot_list.append(labels_onehot)
        labels_onehot_list = torch.stack(labels_onehot_list, dim=0)
        label = labels_onehot_list / labels_onehot_list.sum(dim=1, keepdim=True)

        x = self.pooler(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        pred = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.mean(torch.sum(-label * torch.log(pred), dim=1), dim=0) / 10.0
        return {"multi-label-loss": loss}


# class Global_ROI_Pool(nn.Module):
#     """
#     """
#     def __init__(self, pooled_size=7):
#         self.sum_table = integral_image_forward
#         pivots = torch.from_numpy(
#             np.arange(0, pooled_size, dtype=np.float32)) / pooled_size
#         self.pivots = pivots.view(1, -1)
#         self.end = torch.from_numpy(np.asarray([[-1], [-1]], dtype=np.float32))
#         self.pad_00 = nn.ConstantPad2d((1, 0, 1, 0), 0.0)
#         self.pad_01 = nn.ConstantPad2d((0, 1, 1, 0), 0.0)
#         self.pad_10 = nn.ConstantPad2d((1, 0, 0, 1), 0.0)
    
#     def forward(self, x):
#         x = self.sum_table(x)
#         size = torch.FloatTensor(list(x.size()[2:]))
#         pieces = size.view(-1, 1) * self.pivots
#         pieces = torch.cat((pieces, self.end), dim=1)
#         coords = torch.meshgrid(*torch.round(pieces).to(torch.long))
#         coord_values = x[coords[0], coords[1]]
#         coord_values_00 = self.pad_00(coord_values)
#         coord_values_01 = self.pad_01(coord_values)
#         coord_values_10 = self.pad_10(coord_values)
#         value = coord_values + coord_values_00 - coord_values_01 - coord_values_10
#         return values[:, :, 1:, 1:]


# class Global_ROI_Align(nn.Module):
#     """
#     """
#     def __init__(self, ):
#         self.sum_table = integral_image_forward

#     def forward(self, x):
#         x = self.sum_table(x)


# class Global_ROI_Pool(ROIPool):

#     def __init__(self, pooled_size=7):
#         pivots = torch.from_numpy(
#             np.arange(0, pooled_size, dtype=np.float32)) / pooled_size
#         self.pivots = pivots.view(1, -1)
#         self.end = torch.from_numpy(np.asarray([[-1], [-1]], dtype=np.float32))
    
#     def forward(self, input):
#         size = torch.FloatTensor(list(input.size()[2:]))
#         pieces = size.view(-1, 1) * self.pivots
#         pieces = torch.cat((pieces, self.end), dim=1)
#         x, y = torch.meshgrid(*torch.round(pieces).to(torch.long))
#         x_, y_ = x[1:, 1:], y[1:, 1:]
#         _x, _y = x[:-1, :-1], y[:-1, :-1]
#         input[:, :, _x:x_, _y:y_]


# class Global_ROI_Pool(ROIPool):

#     def __init__(self, output_size=7, spatial_scale=1):
#         super(Global_ROI_Pool).__init__(output_size, spatial_scale)
            
#     def forward(self, x):
#         """
#         Args: x (N,C,H,W) float tensor
#         """
#         x1y1 = torch.FloatTensor(list(input.size()[2:]))
#         x0y0 = torch.zeros_like(x1y1)
#         rois = torch.cat((x0y0, x1y1), dim=0)
#         rois = 


# def integral_image_forward(image):
#     """
#     Args:
#         image: tensor of shape (N, C, H, W) 
#     Return:
#         iimage: tensor of shape (N, C, H, W)
#     """
#     iimage = torch.cumsum(image, dim=2)
#     return torch.cumsum(iimage, dim=3)


def build_multilabel_cls(config):
    return MultiLabelCls(config)
