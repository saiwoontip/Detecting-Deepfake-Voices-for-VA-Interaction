import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# --- 1. GPU CONFIGURATION ---
# Check if a CUDA-enabled GPU is available. If not, fall back to the CPU.
# This is the most critical step for GPU execution.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- 2. PROJECT CONFIGURATION ---
IMAGE_DIR = 'mel_spectrogram_images/'
IMAGE_SIZE = 224  # ResNet models expect 224x224 images
BATCH_SIZE = 32
NUM_EPOCHS = 45 

# --- 3. CUSTOM DATASET  CLASS ---
# This class loads  image paths and labels from your directory structure.
class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Map class names to integer labels
        self.class_to_idx = {'real': 0, 'fake': 1}

        for label_name, label_idx in self.class_to_idx.items():
            class_dir = os.path.join(image_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found at {class_dir}")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Open image and convert to RGB to ensure it has 3 channels for ResNet
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 4. DATA PREPARATION ---
# Define standard transformations for pre-trained models.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(), # Flips images horizontally
        transforms.RandomRotation(10),     # Rotates images by up to 10 degrees
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# Create the full dataset and split it
full_dataset = SpectrogramDataset(IMAGE_DIR, transform=data_transforms['train'])
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

# --- 5. MODEL ARCHITECTURE ---
# Load a pre-trained ResNet-152 model
# New, recommended way
from torchvision.models import ResNet101_Weights
model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

# Freeze all layers in the base model
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification layer for our binary task
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

# **IMPORTANT**: Move the entire model to the selected device (GPU or CPU)
model.to(device)

# --- 6. TRAINING SETUP ---
# Define loss function and optimizer (only for the new, unfrozen layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 7. TRAINING LOOP ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    # Use tqdm for a progress bar
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        # **IMPORTANT**: Move input tensors and labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

print("\n--- Finished Training ---")

# --- 8. EVALUATION ---
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in val_loader:
        # **IMPORTANT**: Move validation inputs to the GPU
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        # Predictions are the output of the sigmoid, rounded to 0 or 1
        preds = (outputs > 0.5).float()
        
        # Move labels and preds back to CPU for sklearn compatibility
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# --- 9. DISPLAY RESULTS ---
print("\n--- Evaluation Results ---")
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()