import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

import random
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from PIL import Image

zip_filename = '/content/test-20260221T085538Z-1-001.zip' 
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
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# Load model weights if available
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            elif 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state, strict=False)
        else:
            raise
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from '{MODEL_PATH}'")
else:
    print(f"Warning: model file '{MODEL_PATH}' not found. Using randomly initialized weights.")
    model = model.to(DEVICE)
    model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Guard: ensure we have predictions
if len(all_labels) == 0:
    print("No samples found in the dataset. Exiting evaluation.")
    raise SystemExit(1)

# ==========================+
# Overall Accuracy
# ==========================+
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ==========================+
# F1 Score
# ==========================+
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# ==========================+
# Per-Class Accuracy
# ==========================+
print("\nPer-Class Accuracy:")
conf_matrix = confusion_matrix(all_labels, all_preds)
class_correct = conf_matrix.diagonal()
class_total = conf_matrix.sum(axis=1)

for i in range(NUM_CLASSES):
    if class_total[i] > 0:
        accuracy = class_correct[i] / class_total[i]
        print(f"  Class {class_names[i]} (Total: {class_total[i]}): {accuracy * 100:.2f}%")
    else:
        print(f"  Class {class_names[i]} (Total: {class_total[i]}): No samples found.")

# Accuracy for class 5
class_5_index = int('5') 
if class_total[class_5_index] > 0:
    accuracy_class_5 = class_correct[class_5_index] / class_total[class_5_index]
    print(f"\nAccuracy for Class 5: {accuracy_class_5 * 100:.2f}%")
else:
    print("\nAccuracy for Class 5: No samples found for this class.")

# ==========================+
# Confusion Matrix Plot
# ==========================+
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ==========================+
# Single Image Prediction Function
# ==========================+
def predict_single_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image '{image_path}': {e}")
        return

    image = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")


# Predict a specific image as requested
specific_image_path = 'data/test/5/340.png'
if os.path.exists(specific_image_path):
    predict_single_image(specific_image_path)
else:
    print(f"\nWarning: Specific image '{specific_image_path}' not found. Skipping prediction for this image.")