import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, 
                             RocCurveDisplay,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# train loop
def training_loop(model, dataloader, loss_fn, optimizer, device):
  model.train()

  train_loss, train_acc = 0, 0

  for batch in dataloader:
    inputs, labels = batch['dce_mri'].to(device), batch['label'].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    probs = torch.sigmoid(outputs)
    predictions = (probs > 0.5).float()
    loss = loss_fn(outputs, labels)
    train_loss += loss.item()

    loss.backward()
    optimizer.step()

    train_acc += (predictions == labels).sum().item()/len(labels)

  # per batch stats
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc

# validation loop
def validation_loop(model, dataloader, loss_fn, device):
  model.eval()

  val_loss, val_acc = 0, 0
  all_preds, all_labels = [], []

  with torch.no_grad():
    for batch in dataloader:
      inputs, labels = batch['dce_mri'].to(device), batch['label'].to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      val_loss += loss.item()

      probs = torch.sigmoid(outputs)
      predictions = (probs > 0.5).float()
      val_acc += (predictions == labels).sum().item() / len(labels)


      all_preds.extend(probs.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    return val_loss, val_acc, all_preds, all_labels


# train function
def train(model, train_dataloader, validation_loader, loss_fn, optimizer, num_epochs, device):

  results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

  for epoch in range(num_epochs):
    train_loss, train_acc = training_loop(model, train_dataloader, loss_fn, optimizer, device)
    val_loss, val_acc, y_prob, y_true = validation_loop(model, validation_loader, loss_fn, device)

    if epoch == num_epochs - 1:

      y_prob = np.array(y_prob).flatten()
      y_true = np.array(y_true).flatten().astype(int)
      y_pred = (y_prob > 0.5).astype(int)

      print("\nClassification Report:\n")
      print(classification_report(y_true, y_pred))

      # Confusion Matrix
      class_names = ["No pCR", "pCR"]
      cm = confusion_matrix(y_true, y_pred)

      cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
      cmd.plot(cmap=plt.cm.Blues, colorbar=True)
      cmd.ax_.set_xlabel("Predicted")
      cmd.ax_.set_ylabel("True")
      cmd.ax_.set_title("Confusion Matrix")

      plt.tight_layout()
      plt.show()

      # ROC Curve
      auc = roc_auc_score(y_true, y_prob)
      RocCurveDisplay.from_predictions(y_true, y_prob)
      plt.title(f"ROC Curve (AUC = {auc:.2f})")
      plt.show()

      # Save Model
      model_path = "pCR_Models"
      os.makedirs(model_path, exists_ok = True) 
      torch.save(model, f'{model_path}/MIA-Classifier.pth')

    print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['val_loss'].append(val_loss)
    results['val_acc'].append(val_acc)
