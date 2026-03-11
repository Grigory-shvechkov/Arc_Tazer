import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data_set"
BATCH_SIZE = 32
IMG_SIZE = 224
MODEL_PATH = "multi_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# TRANSFORMS
# -----------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# -----------------------
# LOAD DATA
# -----------------------
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

classes = val_dataset.classes
num_classes = len(classes)

print(f"\nClasses: {classes}")
print(f"Validation samples: {len(val_dataset)}")

# -----------------------
# LOAD MODEL
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH}")

# -----------------------
# INFERENCE & CONFUSION MATRIX
# -----------------------
print("\n" + "="*50)
print("🔍 ANALYZING VALIDATION SET...")
print("="*50)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✨ Overall Validation Accuracy: {accuracy*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(f"\n📊 CONFUSION MATRIX:")
print(f"{'':15} " + " ".join([f"{c:>10}" for c in classes]))
for i, class_name in enumerate(classes):
    print(f"{class_name:15} {cm[i]}")

# Per-class accuracy
print(f"\n📈 PER-CLASS PERFORMANCE:")
print("-" * 50)
for i, class_name in enumerate(classes):
    class_acc = cm[i, i] / cm[i].sum() * 100
    class_samples = cm[i].sum()
    print(f"{class_name:15} | Accuracy: {class_acc:6.2f}% | Samples: {class_samples:3.0f}")

# Which classes are confused with which?
print(f"\n🔴 MOST CONFUSED PAIRS:")
print("-" * 50)
confusion_pairs = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((cm[i, j], classes[i], classes[j]))

confusion_pairs.sort(reverse=True)
for count, true_class, pred_class in confusion_pairs[:5]:
    print(f"  {int(count):3.0f} images of '{true_class:10}' predicted as '{pred_class:10}'")

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print("-" * 50)
print(classification_report(all_labels, all_preds, target_names=classes))

# Save confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Confusion Matrix - Validation Accuracy: {accuracy*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print(f"\n✅ Confusion matrix saved as 'confusion_matrix.png'")

print("\n" + "="*50)
print("💡 ANALYSIS COMPLETE")
print("="*50)
