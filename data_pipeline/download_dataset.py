
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