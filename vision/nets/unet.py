import torch
from torch import nn
from functools import partial
from collections import OrderedDict


class DownUp(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels,
                 use_bias, inner_layer, outer_most,
                 norm, down_act, up_act,
                 skip_type = 'concat',
                 layer_index=""):
        super(DownUp, self).__init__()
        self.skip_type = skip_type
        self.outer_most = outer_most
        
        down = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm(inner_channels),
            down_act(),
        )
        
        if skip_type == 'concat' and inner_layer is not None:
            inner_channels = inner_channels * 2
                
        up = nn.Sequential(
            nn.ConvTranspose2d(inner_channels, out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
            norm(out_channels),
            up_act()
        )
        
        if inner_layer is not None:
            down_up = OrderedDict([
                (f'down{layer_index}', down),
                (f'inner{layer_index}', inner_layer),
                (f'up{layer_index}', up),
            ])
        else:
            down_up = OrderedDict([
                (f'down{layer_index}', down),
                (f'up{layer_index}', up),
            ])
        self.module_name = f'down_up{layer_index}'
        self.add_module(self.module_name, nn.Sequential(down_up))
            
    def forward(self, x):
        m = self._modules[self.module_name]
        if self.skip_type == 'concat' and not self.outer_most:
            return torch.cat([x, m(x)], 1)
        elif self.skip_type == 'residual' and not self.outer_most:
            return x + m(x)
        else:
            return m(x)


class UNet(nn.Module):

    def __init__(self, channel_config, output_channels=None, 
                 skip_type='concat', 
                 norm=nn.InstanceNorm2d, 
                 down_act=partial(nn.LeakyReLU, 0.2, inplace=True),
                 up_act=partial(nn.ReLU, inplace=True),
                 output_act=nn.Tanh):
        
        super(UNet, self).__init__()
        
        if not output_channels:
            output_channels = channel_config[0]  # output channels equals input channels
        
        if type(norm) == partial:
            use_bias = norm.func == nn.InstanceNorm2d
        else:
            use_bias = norm == nn.InstanceNorm2d
            
        inner_layer = None
        outer_most = False
        for i in range(len(channel_config) - 1, 0, -1):
            in_channels = channel_config[i - 1]
            inner_channels = channel_config[i]
            
            if i == 1 and out_channels:  # for the output layer
                out_channels = output_channels
                outer_most = True  # no skip for the output layer
            else:
                out_channels = in_channels
            inner_layer = DownUp(in_channels, out_channels, inner_channels,
                               use_bias, inner_layer, outer_most,
                               norm, down_act, up_act,
                               skip_type=skip_type,
                               layer_index = i)
            
        self.unet = inner_layer
        
        
    def forward(self, x):
        return self.unet(x)


def unet256(output_channels=3):
    channel_config = [3, 64, 128, 256] + [512] * 5
    return UNet(channel_config, output_channels)


def unet128(output_channels=3):
    channel_config = [3, 64, 128, 256] + [512] * 4
    return UNet(channel_config, output_channels)


def residual_unet128(output_channels=3):
    channel_config = [3, 64, 128, 256] + [512] * 4
    return UNet(channel_config, output_channels, skip_type='residual')


def residual_unet256(output_channels=3):
    channel_config = [3, 64, 128, 256] + [512] * 5
    return UNet(channel_config, output_channels, skip_type='residual')


if __name__ == '__main__':
    input = torch.randn(2, 3, 512, 512)
    u = unet256()
    output = u(input)
    assert input.size() == output.size()

