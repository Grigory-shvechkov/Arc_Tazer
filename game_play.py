import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import mss
import time

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "multi_classifier.pth"  # multi-class saved model
CLASS_NAMES = ['fire', 'no_fire', 'no_shock', 'shock']
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD MODEL + CLASS NAMES
# -----------------------
num_classes = len(CLASS_NAMES)

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded from:", MODEL_PATH)
print("Loaded classes:", CLASS_NAMES)

# -----------------------
# TRANSFORMS
# -----------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)  # shape: [num_classes]
        pred_index = torch.argmax(probs).item()
        pred_label = CLASS_NAMES[pred_index]
        pred_probs = [(CLASS_NAMES[i], round(probs[i].item() * 100, 1)) for i in range(num_classes)]

    return pred_label, pred_index, pred_probs

# -----------------------
# SCREEN CAPTURE SETUP
# -----------------------
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # full screen
sct = mss.mss()

# -----------------------
# DISPLAY WINDOW SETUP
# -----------------------
window_name = "Damage Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 400, 150)  # bigger for multiple classes
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0)

print("Press 'q' to quit")

# -----------------------
# MAIN LOOP
# -----------------------
while True:
    # Capture screen
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)

    # Predict
    pred_label, pred_index, pred_probs = predict_frame(frame)

    # Create blank image for display
    h, w = 150, 400
    display_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Display all class probabilities
    y_offset = 30
    for i, (cls, prob) in enumerate(pred_probs):
        # highlight the highest probability class in red
        color = (0, 0, 255) if i == pred_index else (0, 255, 0)
        text = f"{cls}: {prob}%"
        cv2.putText(display_img, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30  # move down for next class

    # Show small window
    cv2.imshow(window_name, display_img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.03)  # ~30 FPS

cv2.destroyAllWindows()