import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt

import zipfile

zip_filename = 'train-20260221T085534Z-1-001.zip' 
output_directory = 'data'

if os.path.exists(zip_filename) and zip_filename.endswith('.zip'):
    print(f'Unzipping {zip_filename}...')
    os.makedirs(output_directory, exist_ok=True) 
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(output_directory)
    print(f'Successfully unzipped {zip_filename} into {output_directory}')
else:
    print("Not .zip file")

# ==========================
# Configuration
# ==========================
DATA_DIR = "data/train/"
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=False)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Training Loop
# ==========================
train_losses = []
train_accs = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.2f}%")

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "setA.pth")
print("Model saved!")

# ==========================
# Plot and save training curves
# ==========================
os.makedirs("plots", exist_ok=True)

# Loss curve
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join('plots', 'training_loss.png'))
plt.close()

# Accuracy curve
plt.figure()
plt.plot(range(1, len(train_accs) + 1), train_accs, marker='o')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig(os.path.join('plots', 'training_accuracy.png'))
plt.close()

print("Training curves saved to the 'plots' directory")
