
import os
from dotenv import load_dotenv
from roboflow import Roboflow

def main():


    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv('ROBOFLOW_API_KEY')
    workspace = os.getenv('ROBOFLOW_WORKSPACE')
    project = os.getenv('ROBOFLOW_PROJECT')

    print("="*60)
    print("ROBOFLOW CONFIGURATION TEST")
    print("="*60)
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"API Key: {'*' * 20}{api_key[-4:] if api_key else 'NOT FOUND'}")
    print("="*60)

    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file!")

    if not all([api_key, workspace, project]):
        print("\n Missing configuration in .env file!")
        exit(1)


    print("\nConnecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    print(f"Accessing workspace: {workspace}")
    ws = rf.workspace(workspace)

    print(f"Accessing project: {project}")
    proj = ws.project(project)

    print(f"\n✓ Project found: {proj.name}")
    print(f"  Type: {proj.type}")
    print(f"  Total images: 2246")

    print("\nDownloading dataset (version 4)...")
    try:
        dataset = proj.version(1).download("yolov7")

        print(f"\n Success!")
        print(f"Dataset location: {dataset.location}")
        print("\nDataset structure:")

        # Verify files exist
        train_images = os.path.join(dataset.location, "train", "images")
        if os.path.exists(train_images):
            num_train = len(os.listdir(train_images))
            print(f"  Train images: {num_train}")

        valid_images = os.path.join(dataset.location, "valid", "images")
        if os.path.exists(valid_images):
            num_valid = len(os.listdir(valid_images))
            print(f"  Valid images: {num_valid}")

        test_images = os.path.join(dataset.location, "test", "images")
        if os.path.exists(test_images):
            num_test = len(os.listdir(test_images))
            print(f"  Test images: {num_test}")

    except Exception as e:
        print(f"\n Download failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Try downloading again (run this script again)")
        print("3. Try downloading from Roboflow web interface:")
        print(f"   https://app.roboflow.com/{workspace}/{project}/1")
        exit(1)

if __name__ == "__main__":
    main()



# import os
# import shutil
# from pathlib import Path
#
#
# def convert_yolo_to_binary_classification(dataset_path):
#     """
#     Convert YOLOv7 object detection dataset to binary classification
#
#     Args:
#         dataset_path: Path to downloaded Roboflow dataset
#     """
#
#     output_path = Path("rip_current_binary_dataset")
#     output_path.mkdir(exist_ok=True)
#
#     # Create structure
#     for split in ['train', 'valid', 'test']:
#         for class_name in ['positive', 'negative']:
#             (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
#
#     stats = {'train': {'positive': 0, 'negative': 0},
#              'valid': {'positive': 0, 'negative': 0},
#              'test': {'positive': 0, 'negative': 0}}
#
#     for split in ['train', 'valid', 'test']:
#         image_dir = Path(dataset_path) / split / 'images'
#         label_dir = Path(dataset_path) / split / 'labels'
#
#         if not image_dir.exists():
#             print(f"Warning: {image_dir} not found")
#             continue
#
#         for img_file in image_dir.glob('*'):
#             if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
#                 continue
#
#             label_file = label_dir / f"{img_file.stem}.txt"
#
#             # Check if rip current exists
#             has_rip_current = False
#             if label_file.exists():
#                 with open(label_file, 'r') as f:
#                     lines = f.readlines()
#                     has_rip_current = len(lines) > 0
#
#             # Copy to appropriate folder
#             if has_rip_current:
#                 dest_dir = output_path / split / 'positive'
#                 stats[split]['positive'] += 1
#             else:
#                 dest_dir = output_path / split / 'negative'
#                 stats[split]['negative'] += 1
#
#             shutil.copy(img_file, dest_dir / img_file.name)
#
#     # Print statistics
#     print("\n" + "=" * 60)
#     print("DATASET CONVERSION COMPLETE")
#     print("=" * 60)
#
#     total_positive = 0
#     total_negative = 0
#
#     for split in ['train', 'valid', 'test']:
#         pos = stats[split]['positive']
#         neg = stats[split]['negative']
#         total = pos + neg
#         total_positive += pos
#         total_negative += neg
#
#         print(f"\n{split.upper()}:")
#         print(f"  Positive (has rip current): {pos:4d} ({pos / total * 100:.1f}%)")
#         print(f"  Negative (no rip current):  {neg:4d} ({neg / total * 100:.1f}%)")
#         print(f"  Total:                      {total:4d}")
#
#     print(f"\n" + "=" * 60)
#     print(f"GRAND TOTAL:")
#     print(f"  Positive: {total_positive}")
#     print(f"  Negative: {total_negative}")
#     print(f"  Total:    {total_positive + total_negative}")
#     print("=" * 60)
#
#     return output_path
#
#
# # Usage
# dataset_path = dataset.location  # From Step 2
# binary_dataset_path = convert_yolo_to_binary_classification(dataset_path)