import os
import random
import shutil

def split_class(train_dir, val_dir, split_ratio=0.2, move_files=False):
    os.makedirs(val_dir, exist_ok=True)

    images = [
        f for f in os.listdir(train_dir)
        if os.path.isfile(os.path.join(train_dir, f))
    ]

    if len(images) == 0:
        print(f"Skipping {train_dir} (no images found)")
        return

    random.shuffle(images)

    split_count = int(len(images) * split_ratio)
    val_images = images[:split_count]

    for img in val_images:
        src = os.path.join(train_dir, img)
        dst = os.path.join(val_dir, img)

        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy(src, dst)

    print(f"{os.path.basename(train_dir)} → moved {len(val_images)} files")


def split_dataset(base_path="data_set", split_ratio=0.2, seed=42):
    random.seed(seed)

    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    os.makedirs(val_path, exist_ok=True)

    # Automatically detect all class folders
    for cls in os.listdir(train_path):
        train_dir = os.path.join(train_path, cls)

        if os.path.isdir(train_dir):
            val_dir = os.path.join(val_path, cls)
            split_class(train_dir, val_dir, split_ratio)


if __name__ == "__main__":
    split_dataset(split_ratio=0.2)