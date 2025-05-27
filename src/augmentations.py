import torch
import torch.nn as nn
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomGaussianBlur,
    RandomGaussianNoise,
    ColorJitter,
    RandomGamma,
    AugmentationSequential
)
from kornia.constants import Resample

class BitemporalAugmentationModule(nn.Module):
    """
    Augmentation module for bitemporal change detection.
    Applies:
    - Shared geometric augmentations to all channels + mask
    - Independent radiometric augmentations to PAN and RGB parts
    """
    def __init__(self):
        super().__init__()

        # Geometric transformations (shared across image + mask)
        self.geom_aug = AugmentationSequential(
            RandomHorizontalFlip(p=0.5, same_on_batch=True),
            RandomVerticalFlip(p=0.5, same_on_batch=True),
            RandomRotation90(times=(0, 3), resample=Resample.NEAREST, p=0.5, same_on_batch=True),
            data_keys=["image", "mask"]
        )

        # Radiometric augmentations for PAN (channel 0)
        self.pan_aug = AugmentationSequential(
            # RandomGamma(gamma=(0.8, 1.2), p=0.2),
            # RandomGaussianNoise(mean=0.0, std=0.05, p=0.2),
            data_keys=["image"]
        )

        # Radiometric augmentations for RGB (channels 1–3)
        self.rgb_aug = AugmentationSequential(
            # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=1.0),
            # RandomGamma(gamma=(0.7, 1.5), p=0.2),
            # RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
            data_keys=["image"]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x: [B, 4, H, W], y: [B, 1, H, W]

        # Shared geometric aug
        x, y = self.geom_aug(x, y)

        # Split image and normalize 
        pan = x[:, 0:1] / 255.0
        rgb = x[:, 1:4] / 255.0

        # Independent aug
        pan = self.pan_aug(pan)
        rgb = self.rgb_aug(rgb)

        # Restore scale
        x_aug = torch.cat([pan * 255.0, rgb * 255.0], dim=1)
        return x_aug, y
