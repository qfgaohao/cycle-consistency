from .unet import UNet, unet256, residual_unet128, residual_unet256
from .discriminator import PatchDiscriminator, patch_discriminator8
from .cycle_gan import (
    CycleGan, CycleGanOptimizer, simple_adam_optimizer,
    unet128_cycle_gan, unet256_cycle_gan,
    residual_unet128_cycle_gan, residual_unet256_cycle_gan,
    resnet_cycle_gan
)
from .resnet import resnet