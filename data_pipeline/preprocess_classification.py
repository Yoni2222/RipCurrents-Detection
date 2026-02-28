"""
Convert YOLO format to binary classification

"""
import os
import shutil
from pathlib import Path


def convert_yolo_to_binary_classification(dataset_path):
    """
    Convert YOLOv7 object detection dataset to binary classification

    Args:
        dataset_path: Path to downloaded Roboflow dataset
    """
    output_path = Path("../rip_current_binary_dataset")
    output_path.mkdir(exist_ok=True)

    # Create structure
    for split in ['train', 'valid', 'test']:
        for class_name in ['positive', 'negative']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    stats = {
        'train': {'positive': 0, 'negative': 0},
        'valid': {'positive': 0, 'negative': 0},
        'test': {'positive': 0, 'negative': 0}
    }

    for split in ['train', 'valid', 'test']:
        image_dir = Path(dataset_path) / split / 'images'
        label_dir = Path(dataset_path) / split / 'labels'

        if not image_dir.exists():
            print(f"Warning: {image_dir} not found")
            continue

        for img_file in image_dir.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            label_file = label_dir / f"{img_file.stem}.txt"

            # Check if rip current exists
            has_rip_current = False
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    has_rip_current = len(lines) > 0

            # Copy to appropriate folder
            if has_rip_current:
                dest_dir = output_path / split / 'positive'
                stats[split]['positive'] += 1
            else:
                dest_dir = output_path / split / 'negative'
                stats[split]['negative'] += 1

            shutil.copy(img_file, dest_dir / img_file.name)

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET CONVERSION COMPLETE")
    print("=" * 60)

    total_positive = 0
    total_negative = 0

    for split in ['train', 'valid', 'test']:
        pos = stats[split]['positive']
        neg = stats[split]['negative']
        total = pos + neg
        total_positive += pos
        total_negative += neg

        print(f"\n{split.upper()}:")
        print(f"  Positive (has rip current): {pos:4d} ({pos / total * 100:.1f}%)")
        print(f"  Negative (no rip current):  {neg:4d} ({neg / total * 100:.1f}%)")
        print(f"  Total:                      {total:4d}")

    print(f"\n" + "=" * 60)
    print(f"GRAND TOTAL:")
    print(f"  Positive: {total_positive}")
    print(f"  Negative: {total_negative}")
    print(f"  Total:    {total_positive + total_negative}")
    print("=" * 60)

    return output_path


def main():
    # Find the downloaded dataset
    dataset_path = "../Rip-Current-Monitoring-1"

    if not os.path.exists(dataset_path):
        print(f" Dataset not found at: {dataset_path}")
        print("Please run download_dataset.py first")
        print("Or update the dataset_path variable in this file")
        return

    print(f"Converting dataset from: {dataset_path}")
    binary_dataset_path = convert_yolo_to_binary_classification(dataset_path)

    print(f"\n Binary dataset created at: {binary_dataset_path}")
    print(f"\nNext step: Run train_classification.py")


if __name__ == "__main__":
    main()