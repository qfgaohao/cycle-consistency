import torch
from torch import nn
from functools import partial
from collections import OrderedDict


def conv(in_channels, out_channels, stride, norm, act, use_bias):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                             stride=stride, padding=1, bias=use_bias),
            norm(out_channels),
            act(),
    )


class PatchDiscriminator(nn.Module):

    def __init__(self, layer_config, norm=nn.BatchNorm2d, 
                 act=partial(nn.LeakyReLU, 0.2, inplace=True)):
        super(PatchDiscriminator, self).__init__()
        if type(norm) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm.func == nn.InstanceNorm2d
        else:
            use_bias = norm == nn.InstanceNorm2d
        layers = OrderedDict()
        for i, (in_channels, out_channels, stride) in enumerate(layer_config[:-1]):
            layers[f"conv{i}"] = conv(in_channels, out_channels, stride, norm, act, use_bias)
        in_channels, out_channels, stride = layer_config[-1]
        layers[f"conv{len(layer_config) - 1}"] = nn.Conv2d(in_channels, out_channels,
         kernel_size=4, stride=stride, padding=1)

        self.net = nn.Sequential(layers)

    def forward(self, input):
        return self.net(input)


def patch_discriminator8(in_channels=3):
    layer_config = [
        # in_channels, out_channels, stride
        (in_channels, 64, 2),
        (64, 128, 2),
        (128, 256, 2),
        (256, 512, 1),
        (512, 1, 1)
    ]
    return PatchDiscriminator(layer_config)


if __name__ == "__main__":
    input = torch.randn(2, 3, 512, 512)
    net = patch_discriminator8()
    output = net(input)
    assert tuple(output.size()) == (2, 1, 62, 62)