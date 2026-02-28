"""
YOLO Object Detection Training Engine
Handles: dataset preprocessing, YOLO training, and model saving.
Supports: RGB, REPLACEMENT, HSV_WAVELET, DIRECTIONAL, FUSION
"""

import os
import shutil
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from data_pipeline.preprocess_detection import process_image


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"\n  SEED SET TO: {seed}")


def build_processed_dataset(src_root, dst_root, model_type):
    """
    Preprocess images with wavelet transforms and build a YOLO-format dataset.

    Args:
        src_root: Path to original Roboflow dataset (with train/valid/test splits)
        dst_root: Path where the processed dataset will be created
        model_type: One of 'RGB', 'REPLACEMENT', 'HSV_WAVELET', 'DIRECTIONAL', 'FUSION'
    """
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    splits = ['train', 'valid', 'test']
    print(f"\n  Building {model_type} dataset at: {dst_root}")

    for split in splits:
        (dst_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_root / split / 'labels').mkdir(parents=True, exist_ok=True)

        src_img_dir = src_root / split / 'images'
        src_lbl_dir = src_root / split / 'labels'

        if not src_img_dir.exists():
            print(f"  Skipping {split} (directory not found)")
            continue

        images = list(src_img_dir.glob('*.*'))
        print(f"  Processing {split}: {len(images)} images")

        for img_path in tqdm(images, desc=f"  {split}"):
            processed_img = process_image(img_path, model_type)

            if processed_img is not None:
                cv2.imwrite(str(dst_root / split / 'images' / img_path.name), processed_img)

                # Copy matching label file
                label_name = f"{img_path.stem}.txt"
                src_label = src_lbl_dir / label_name
                if src_label.exists():
                    shutil.copy(src_label, dst_root / split / 'labels' / label_name)

    # Create data.yaml for YOLO
    yaml_data = {
        'path': str(dst_root.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['rip_current']
    }

    yaml_path = dst_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    print(f"  Dataset ready! data.yaml saved at: {yaml_path}")
    return yaml_path


# Need cv2 for imwrite in build_processed_dataset



def train_detection(args, device):
    """
    Main YOLO detection training pipeline.

    Args:
        args: Parsed arguments containing:
            - detection_model_type: RGB, REPLACEMENT, HSV_WAVELET, DIRECTIONAL, FUSION
            - yolo_dataset_path: Path to original Roboflow YOLO dataset
            - yolo_model_size: 'n' (nano), 's' (small), 'm' (medium), etc.
            - epochs, batch_size, patience, seed
        device: torch device
    """
    model_type = args.detection_model_type.upper()
    print("=" * 50)
    print(f"  YOLO Detection Training")
    print(f"  Model Type: {model_type}")
    print(f"  Device: {device}")
    print("=" * 50)

    # Set seed
    set_seed(args.seed)

    # Paths
    src_dataset = Path(args.yolo_dataset_path)
    if not src_dataset.exists():
        print(f"\n  ERROR: Dataset not found at {src_dataset}")
        print(f"  Please download the Roboflow dataset first using:")
        print(f"    python data_pipeline/download_dataset.py")
        return

    # Create processed dataset
    processed_dir = Path(f'./processed_datasets/dataset_{model_type.lower()}')
    yaml_path = build_processed_dataset(src_dataset, processed_dir, model_type)

    # Initialize YOLO model
    yolo_size = getattr(args, 'yolo_model_size', 'n')
    model_name = f'yolov8{yolo_size}.pt'
    print(f"\n  Loading base model: {model_name}")
    model = YOLO(model_name)

    # Output directory
    save_dir = Path(f'./runs/detection/{model_type.lower()}')
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = f'{model_type.lower()}_seed{args.seed}'
    # Train
    print(f"\n  Starting training for {model_type}...")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        patience=args.patience,
        project='runs/detection',
        name=f'{model_type.lower()}',
        device=str(device),
        seed=args.seed,
    )

    # Copy best model to saved_models
    best_src = Path('runs/detection') / run_name / 'weights' / 'best.pt'
    if best_src.exists():
        saved_models_dir = Path('saved_models')
        saved_models_dir.mkdir(exist_ok=True)
        best_dst = saved_models_dir / f'best_yolo_{model_type.lower()}.pt'
        shutil.copy(best_src, best_dst)
        print(f"\n  Best model copied to: {best_dst}")
    print(f"\n  Training complete! Results in: {save_dir}")
    print("=" * 50)