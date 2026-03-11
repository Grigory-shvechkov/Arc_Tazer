import cv2
import os


video_path = "input.mp4"

project_root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_root, "image_cap")
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        filename = os.path.join(output_dir,f"capture_{frame_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        
    elif key == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()