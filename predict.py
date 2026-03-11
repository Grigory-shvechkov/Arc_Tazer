import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "damage_classifier.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_PREDICT = "data_set/val"   # can point to ANY folder of images
OUTPUT_CSV = "predictions.csv"

print("Using device:", DEVICE)

# -----------------------
# LOAD MODEL + CLASS NAMES
# -----------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

print("Loaded classes:", class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# -----------------------
# TRANSFORMS
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        pred_label = class_names[pred_index]

        # Convert probabilities to readable dict
        prob_dict = {
            class_names[i]: round(probs[0, i].item() * 100, 2)
            for i in range(num_classes)
        }

    return pred_label, prob_dict

# -----------------------
# GATHER ALL IMAGES (recursive)
# -----------------------
image_paths = []

for root, dirs, files in os.walk(FOLDER_TO_PREDICT):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images.")

# -----------------------
# BATCH PREDICTION
# -----------------------
results = []

for path in image_paths:
    filename = os.path.basename(path)
    pred_label, prob_dict = predict(path)

    row = [filename, pred_label]
    for cls in class_names:
        row.append(prob_dict[cls])

    results.append(row)

    print(f"\n{filename}")
    print(f"Predicted: {pred_label}")
    for cls in class_names:
        print(f"   {cls}: {prob_dict[cls]}%")

# -----------------------
# SAVE TO CSV
# -----------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    header = ["filename", "predicted"] + [f"prob_{cls} (%)" for cls in class_names]
    writer.writerow(header)
    writer.writerows(results)

print(f"\nPredictions saved to {OUTPUT_CSV}")