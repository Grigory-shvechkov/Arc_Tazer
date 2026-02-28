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
MODEL_PATH = "shock_classifier.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
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
def predict_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = "SHOCK" if pred_class == 1 else "NO SHOCK"

    return pred_label

# -----------------------
# SCREEN CAPTURE SETUP
# -----------------------
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # full screen
sct = mss.mss()

# -----------------------
# TINY DISPLAY WINDOW SETUP
# -----------------------
window_name = "Shock Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 200, 100)       # initial size
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # always on top (optional)
cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0) # allow resizing

print("Press 'q' to quit")

while True:
    # Capture screen
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)

    # Predict
    label = predict_frame(frame)

    # Create a blank image for display
    h, w = 100, 200
    display_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Put label on it
    color = (0, 0, 255) if label == "SHOCK" else (0, 255, 0)
    cv2.putText(display_img, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show small window
    cv2.imshow(window_name, display_img)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.03)  # ~30 FPSq

cv2.destroyAllWindows()