import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes=50, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Replace RGB input with the single-channel mel spectrogram.
    original_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    # Preserve the pretrained first-layer response when RGB inputs
    # are represented by one repeated channel.
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(
                original_conv.weight.sum(dim=1, keepdim=True)
            )

    model.conv1 = new_conv

    # ESC-50 has 50 target classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model