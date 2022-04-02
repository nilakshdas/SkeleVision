from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_block(in_channels, layer_kwargs: Dict[str, int]) -> nn.Module:
    conv_layer = nn.Conv2d(in_channels, **layer_kwargs)  # type:ignore
    nn.init.kaiming_normal_(conv_layer.weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(conv_layer.bias, 0)  # type:ignore
    return nn.Sequential(conv_layer, nn.ReLU(inplace=True))


def _make_deconv_block(in_channels, layer_kwargs: Dict[str, int]) -> nn.Module:
    deconv_layer = nn.ConvTranspose2d(in_channels, **layer_kwargs)  # type:ignore
    nn.init.kaiming_normal_(deconv_layer.weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(deconv_layer.bias, 0)  # type:ignore
    return deconv_layer


class ModularConvDeconvInterpKeypointHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_kwargs: List[Dict[str, int]],
        deconv_kwargs: List[Dict[str, int]],
        output_size: Tuple[int, int],
    ):
        super().__init__()

        assert len(conv_kwargs) > 0
        assert len(output_size) == 2

        conv_blocks: List[nn.Module] = list()
        for conv_kwarg_dict in conv_kwargs:
            kwargs = dict(kernel_size=3, stride=1, padding=1)
            kwargs.update(conv_kwarg_dict)
            conv_blocks.append(_make_conv_block(in_channels, kwargs))
            in_channels = int(kwargs.pop("out_channels"))  # type: ignore

        deconv_blocks: List[nn.Module] = list()
        for deconv_kwarg_dict in deconv_kwargs:
            kwargs = dict(kernel_size=8, stride=2, padding=1)
            kwargs.update(deconv_kwarg_dict)
            deconv_blocks.append(_make_deconv_block(in_channels, kwargs))
            in_channels = int(kwargs.pop("out_channels"))  # type: ignore

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.deconv_blocks(x)
        return F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )
