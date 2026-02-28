import os
import random
import shutil

def split_class(train_dir, val_dir, split_ratio=0.2):
    os.makedirs(val_dir, exist_ok=True)

    images = [f for f in os.listdir(train_dir)
              if os.path.isfile(os.path.join(train_dir, f))]

    random.shuffle(images)

    split_count = int(len(images) * split_ratio)
    val_images = images[:split_count]

    for img in val_images:
        src = os.path.join(train_dir, img)
        dst = os.path.join(val_dir, img)
        shutil.move(src, dst)

    print(f"Moved {len(val_images)} files from {train_dir} to {val_dir}")


def split_dataset(base_path="data_set", split_ratio=0.2):
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    classes = ["shock", "no_shock"]

    for cls in classes:
        train_dir = os.path.join(train_path, cls)
        val_dir = os.path.join(val_path, cls)
        split_class(train_dir, val_dir, split_ratio)


if __name__ == "__main__":
    split_dataset(split_ratio=0.2)