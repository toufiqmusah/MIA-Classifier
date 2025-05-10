# models.py

import torch
import torch.nn as nn
from nnunet_mednext import create_mednext_v1
from monai.networks.nets import EfficientNetBN, ResNet

# MedNext Classifier Model Class

class MedNextClassifier(nn.Module):
    def __init__(self, in_channels=3, model_id='M', kernel_size=3, encoder_output=256, num_classes=1):
        super(MedNextClassifier, self).__init__()

        self.encoder = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=num_classes,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=True
        )
        self.encoder.eval()

        # to freeze encoder
        for param in self.encoder.parameters():
           param.requires_grad = False

        # downstream classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_output, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
      x = self.encoder.stem(x)
      x = self.encoder.enc_block_0(x)
      x = self.encoder.down_0(x)
      x = self.encoder.enc_block_1(x)
      x = self.encoder.down_1(x)
      x = self.encoder.enc_block_2(x)
      x = self.encoder.down_2(x)
      x = self.encoder.enc_block_3(x)

      return self.classifier(x)
    

# EfficientNet Model Class

class EfficientNetBN3D(nn.Module):
  def __init__(self, spatial_dims = 3, in_channels = 3, model_name: str = 'efficientnet-b4', norm = 'batch', pretrained = False):
    super(EfficientNetBN3D, self).__init__()

    self.model = EfficientNetBN(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        model_name=model_name,
        norm = norm,
        pretrained = pretrained
    )

    self.linear = nn.Linear(1000, 1)
    self.final_activation = nn.Sigmoid()

  def forward(self, x):
    x = self.model(x)
    x = self.linear(x)

    return self.final_activation(x)
  


# ResNet Model Class

class ResNet3D(nn.Module):
  pass