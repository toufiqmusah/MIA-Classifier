import torch
import torch.nn as nn
from collections import OrderedDict

from train import train
from models import MedNextClassifier
from data import train_loader, validation_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('', map_location=device, weights_only=False)

# strip 'model.' prefix
clean_state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    if k.startswith('model.'):
        k = k[len('model.'):]
    clean_state_dict[k] = v

# Init and load
model = MedNextClassifier(in_channels=3, num_classes=2)
model.encoder.load_state_dict(clean_state_dict, strict=False)

# params
loss_criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# train
train(model, train_loader, validation_loader, loss_criterion, optimizer, 5, device)