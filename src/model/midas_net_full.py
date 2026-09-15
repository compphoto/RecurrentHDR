"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
)


def _make_fusion_block(features, use_bn, scale=2):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        scale=scale,
    )


class MidasNet(BaseModel):
    """Network for monocular depth estimation."""

    def __init__(
        self,
        path=None,
        features=256,
        in_c=3,
        out_c=3,
        non_negative=False,
        use_bn=False,
        backbone="resnext101_wsl",
        refine="midas",
        last_residual=False,
    ):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        self.backbone = backbone
        self.scales = [2, 2, 2, 2]
        self.last_res = last_residual

        use_pretrained = True  # False if path is None else True

        # self.pretrained, self.scratch = _make_encoder(backbone="efficientnet_lite3", features=features, use_pretrained=use_pretrained)
        self.pretrained, self.scratch = _make_encoder(
            backbone=self.backbone,
            features=features,
            in_chan=in_c,
            use_pretrained=use_pretrained,
        )
        if self.backbone == "swinv2_b4_384":
            self.scales = [2, 4, 4, 2]

        # MiDaS refinement
        if refine == "midas":
            self.scratch.refinenet4 = FeatureFusionBlock(features, self.scales[0])
            self.scratch.refinenet3 = FeatureFusionBlock(features, self.scales[1])
            self.scratch.refinenet2 = FeatureFusionBlock(features, self.scales[2])
            self.scratch.refinenet1 = FeatureFusionBlock(features, self.scales[3])

        # DPT refinement
        else:
            self.scratch.refinenet1 = _make_fusion_block(
                features, use_bn, self.scales[0]
            )
            self.scratch.refinenet2 = _make_fusion_block(
                features, use_bn, self.scales[1]
            )
            self.scratch.refinenet3 = _make_fusion_block(
                features, use_bn, self.scales[2]
            )
            self.scratch.refinenet4 = _make_fusion_block(
                features, use_bn, self.scales[3]
            )

        # self.scratch.output_conv = nn.Sequential(
        #     nn.Conv2d(features, 256, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, out_c, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        # )

        res_dim = 256 + (in_c if self.last_res else 0)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 256, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # out = self.scratch.output_conv(path_1)
        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out.contiguous())

        if self.last_res:
            out = torch.cat((out, x), dim=1)

        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)

        return out
