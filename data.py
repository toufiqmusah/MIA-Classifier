# data.py

import os
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import monai
from monai.transforms import (
    Compose,
    Invertd,
    Spacingd,
    RandFlipd,
    LoadImaged,
    AsDiscrete,
    Activations,
    AsDiscreted,
    EnsureTyped,
    DeleteItemsd,
    ConcatItemsd,
    Activationsd,
    Orientationd,
    DivisiblePadD,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd)

# custom dataset, dataloader

class MIAClassificationDataset(Dataset):
  def __init__(self, root_dir, clinical_data, transform=None):
    self.root_dir = root_dir
    self.clinical_data = pd.read_csv(clinical_data).dropna(subset=['pcr'])
    self.transform = transform

  def __len__(self):
    return len(self.clinical_data['pcr'])

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    row = self.clinical_data.iloc[idx]
    patient_id = row['patient_id']
    label = torch.tensor([row["pcr"]], dtype=torch.float32)

    img_paths = [
        os.path.join(self.root_dir, patient_id, f'{patient_id}_0000.nii.gz'),
        os.path.join(self.root_dir, patient_id, f'{patient_id}_0001.nii.gz'),
        os.path.join(self.root_dir, patient_id, f'{patient_id}_0002.nii.gz'),
    ]

    data = {"dce_mri": img_paths, "label": label}

    if self.transform is not None:
      data = self.transform(data)

    return data
 

transforms = Compose([
    LoadImaged(keys="dce_mri", reader="ITKReader"),
    EnsureChannelFirstd(keys="dce_mri"),
    Orientationd(keys='dce_mri', axcodes='RAS'),
    # Spacingd(keys='dce_mri', pixdim=(1.0,1.0,1.0), mode='bilinear'),
    ScaleIntensityRangePercentilesd(
         keys=["dce_mri"],
         lower=0,
         upper=99.5,
         b_min=0.0,
         b_max=1.0),
   EnsureTyped(keys='dce_mri'),
   ]
)

dataset = MIAClassificationDataset(root_dir = '', clinical_data = '', transform = transforms)
seed = torch.Generator().manual_seed(42)
train_data, validation_data = random_split(dataset, [0.90, 0.10], seed)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)