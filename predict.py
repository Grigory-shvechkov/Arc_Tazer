import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "shock_classifier.pth"  # Your .pth file
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLDER_TO_PREDICT = "data_set/val/no_shock"  # Change to any folder of images
OUTPUT_CSV = "predictions.csv"

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # shock / no_shock
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
        probs = torch.softmax(output, dim=1)  # tensor stays
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = "shock" if pred_class == 1 else "no_shock"
        prob_no_shock = round(probs[0,0].item() * 100, 2)
        prob_shock    = round(probs[0,1].item() * 100, 2)

    return pred_label, prob_no_shock, prob_shock

# -----------------------
# BATCH PREDICTION
# -----------------------
results = []
for file in os.listdir(FOLDER_TO_PREDICT):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(FOLDER_TO_PREDICT, file)
        label, no_shock_pct, shock_pct = predict(path)
        results.append([file, label, no_shock_pct, shock_pct])
        print(f"{file}: {label}, no_shock {no_shock_pct}%, shock {shock_pct}%")

# -----------------------
# SAVE TO CSV
# -----------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "predicted", "prob_no_shock (%)", "prob_shock (%)"])
    writer.writerows(results)

print(f"\nPredictions saved to {OUTPUT_CSV}")