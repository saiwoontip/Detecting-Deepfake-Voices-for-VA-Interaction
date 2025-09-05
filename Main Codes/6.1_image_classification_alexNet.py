import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================
# --- Data Settings ---
IMAGE_DIR = 'mel_spectrogram_images/'
IMAGE_SIZE = 224 # AlexNet expects 224x224 images

# --- Training Settings ---
BATCH_SIZE = 32
NUM_EPOCHS = 45
LEARNING_RATE = 0.001

# --- GPU Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")


# ====================================================================
# --- 2. DATASET AND DATALOADERS ---
# ====================================================================
class SpectrogramDataset(Dataset):
    """Custom PyTorch Dataset for loading spectrogram images."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'real': 0, 'fake': 1}

        for label_name, label_idx in self.class_to_idx.items():
            class_dir = os.path.join(image_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found at {class_dir}")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Define data transformations and augmentation for training ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Create Datasets and split into training/validation ---
full_dataset = SpectrogramDataset(IMAGE_DIR, transform=data_transforms['train'])
generator = torch.Generator().manual_seed(42) # for reproducible splits
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=generator
)

# Apply validation transforms to the validation set
val_dataset.dataset.transform = data_transforms['val']

# --- Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")


# ====================================================================
# --- 3. MODEL DEFINITION (ALEXNET) ---
# ====================================================================
# Load pre-trained AlexNet model
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Freeze the parameters of the feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier with a new one for our binary task
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

# Move the model to the configured device (GPU or CPU)
model.to(device)

# ====================================================================
# --- 4. TRAINING SETUP ---
# ====================================================================
criterion = nn.BCELoss()
# IMPORTANT: Use model.classifier.parameters() for AlexNet
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# ====================================================================
# --- 5. TRAINING LOOP ---
# ====================================================================
print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

print("\n--- Finished Training ---")


# ====================================================================
# --- 6. EVALUATION ---
# ====================================================================
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy().flatten())

print("\n--- Evaluation Results ---")
target_names = ['Real', 'Fake']
print(classification_report(all_labels, all_preds, target_names=target_names))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()