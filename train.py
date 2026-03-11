import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data_set"
BATCH_SIZE = 32
EPOCHS = 20           # Increased epochs since we lowered the learning rate
IMG_SIZE = 224
LR = 0.00005          # Reduced learning rate to prevent overfitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print(f"🚀 INITIALIZING TRAINING ENVIRONMENT")
print("=" * 50)
print(f"Using device: {device}")

# -----------------------
# TRANSFORMS (Improved for Transfer Learning)
# -----------------------
# ImageNet standard normalization values (Required for ResNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# -----------------------
# LOAD DATA
# -----------------------
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

classes = train_dataset.classes
num_classes = len(classes)

print(f"Classes detected ({num_classes}): {classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------
# HANDLE IMBALANCED DATA
# -----------------------
target_list = torch.tensor(train_dataset.targets)
class_counts = torch.bincount(target_list).float()
total_samples = len(target_list)

class_weights = total_samples / (num_classes * class_counts)
class_weights = class_weights.to(device)

print("\n📊 DATA BALANCE SUMMARY:")
print("-" * 50)
for i, class_name in enumerate(classes):
    print(f" - {class_name.ljust(15)} : {int(class_counts[i])} samples | Assigned Weight: {class_weights[i]:.4f}")
print("-" * 50)

# -----------------------
# MODEL (Transfer Learning)
# -----------------------
model = models.resnet18(weights="IMAGENET1K_V1")

# Add dropout before final layer to reduce overfitting
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# -----------------------
# TRAIN LOOP
# -----------------------
print("\n🔥 STARTING TRAINING LOOP...")
print("=" * 50)

best_val_acc = 0.0
patience_counter = 0
patience_limit = 5

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # -------- TRAINING --------
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    epoch_time = time.time() - start_time

    # -------- LEARNING RATE SCHEDULING --------
    scheduler.step(val_acc)

    # -------- BEST MODEL TRACKING --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "multi_classifier.pth")
        print(f"\n🟢 Epoch [{epoch+1}/{EPOCHS}] completed in {epoch_time:.0f}s")
        print(f"   Train => Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Valid => Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"   ✨ NEW BEST MODEL SAVED! (Validation Acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f"\n🟡 Epoch [{epoch+1}/{EPOCHS}] completed in {epoch_time:.0f}s")
        print(f"   Train => Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Valid => Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"   (No improvement. Patience: {patience_counter}/{patience_limit})")
    
    # -------- EARLY STOPPING --------
    if patience_counter >= patience_limit:
        print(f"\n🛑 EARLY STOPPING TRIGGERED after epoch {epoch+1}")
        print(f"   Best validation accuracy achieved: {best_val_acc:.2f}%")
        break

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE.")
print("=" * 50)
print(f"Model saved as 'multi_classifier.pth' with best validation accuracy: {best_val_acc:.2f}%")