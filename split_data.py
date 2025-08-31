import os
import shutil
import random
from pathlib import Path

RANDOM_SEED = 42

def gather_images(class_dir):
    exts = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    return [p for p in Path(class_dir).rglob('*') if p.suffix.lower() in exts]

def split_and_copy(class_name, src_dir, dst_base, train_ratio=0.8, val_ratio=0.1):
    imgs = gather_images(os.path.join(src_dir, class_name))
    imgs = sorted(imgs)
    if not imgs:
        print(f"[WARN] No images found for class '{class_name}' in {src_dir}.")
        return
    random.Random(RANDOM_SEED).shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        'train': imgs[:n_train],
        'val': imgs[n_train:n_train+n_val],
        'test': imgs[n_train+n_val:]
    }
    for split, paths in splits.items():
        out_dir = Path(dst_base)/split/class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, out_dir/p.name)
    print(f"[OK] {class_name}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

def main():
    raw_base = 'data_raw'  # put two folders inside: with_mask, without_mask
    dst_base = 'data'
    classes = ['with_mask','without_mask']
    for c in classes:
        split_and_copy(c, raw_base, dst_base)

if __name__ == "__main__":
    main()
