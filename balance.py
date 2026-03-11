import os
from collections import Counter

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data_set"  # adjust if your dataset folder is elsewhere
SUBFOLDERS = ["train", "val"]  # dataset splits

# -----------------------
# FUNCTION
# -----------------------
def examine_dataset(data_dir, subfolders):
    print(f"Examining dataset in: {data_dir}\n")
    for split in subfolders:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"⚠️  {split_path} does not exist, skipping")
            continue

        class_counts = {}
        total_images = 0

        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                class_counts[class_name] = num_images
                total_images += num_images

        print(f"--- {split.upper()} ---")
        print(f"Classes: {sorted(class_counts.keys())}")
        print(f"Total images: {total_images}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} images")
        
        # check for imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        if max_count / max(min_count, 1) > 2:  # arbitrary threshold for imbalance
            print("⚠️  Possible class imbalance detected")
        print()

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    examine_dataset(DATA_DIR, SUBFOLDERS)