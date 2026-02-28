"""
YOLO Object Detection Testing & Evaluation Engine
Includes: YOLO evaluation, SAM refinement pipeline, and visual evidence generation.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

from data_pipeline.preprocess_detection import process_image


# ==========================================
# Helper Functions
# ==========================================

def get_box_from_mask(mask):
    """Extract bounding box [x1, y1, x2, y2] from a binary mask."""
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return None
    return [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def load_ground_truth_boxes(label_file, img_w, img_h):
    """Load YOLO-format labels and convert to [x1, y1, x2, y2] pixel coordinates."""
    gt_boxes = []
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, x_c, y_c, bw, bh = map(float, parts[:5])
                x1 = int((x_c - bw / 2) * img_w)
                y1 = int((y_c - bh / 2) * img_h)
                x2 = int((x_c + bw / 2) * img_w)
                y2 = int((y_c + bh / 2) * img_h)
                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def load_sam_model(device):
    """Load SAM model. Downloads weights if not present."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("  ERROR: segment-anything not installed.")
        print("  Install with: pip install segment-anything")
        print("  Also download weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return None

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print(f"  SAM weights not found at: {sam_checkpoint}")
        print(f"  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return None

    print("  Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("  SAM loaded!")
    return predictor


# ==========================================
# Visualization
# ==========================================

def save_visualization(orig_rgb, gt_box, yolo_box, refined_box, sam_mask,
                       iou_yolo, iou_sam, save_path):
    """Save a comparison visualization: GT vs YOLO vs SAM."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(orig_rgb)

    # Ground Truth (Green)
    gx1, gy1, gx2, gy2 = gt_box
    rect_gt = patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth')
    ax.add_patch(rect_gt)

    # YOLO (Red dashed)
    yx1, yy1, yx2, yy2 = yolo_box
    rect_yolo = patches.Rectangle((yx1, yy1), yx2 - yx1, yy2 - yy1,
                                  linewidth=2, edgecolor='red', facecolor='none',
                                  linestyle='--', label='YOLO Detection')
    ax.add_patch(rect_yolo)

    # SAM Refined (Cyan)
    if refined_box is not None:
        sx1, sy1, sx2, sy2 = refined_box
        rect_sam = patches.Rectangle((sx1, sy1), sx2 - sx1, sy2 - sy1,
                                     linewidth=2, edgecolor='cyan', facecolor='none',
                                     label='SAM Refined')
        ax.add_patch(rect_sam)

    # SAM Mask overlay
    if sam_mask is not None:
        masked_image = np.ma.masked_where(sam_mask == 0, sam_mask)
        ax.imshow(masked_image, cmap='jet', alpha=0.4)

    plt.title(f"IoU: YOLO={iou_yolo:.2f} -> SAM={iou_sam:.2f}",
              fontsize=12, color='white', backgroundcolor='black')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.savefig(str(save_path), bbox_inches='tight', dpi=100)
    plt.close(fig)


# ==========================================
# Main Test Functions
# ==========================================

def run_yolo_validation(model, processed_dataset_dir):
    """Run official YOLO validation metrics on the test set."""
    yaml_path = processed_dataset_dir / 'data.yaml'
    if not yaml_path.exists():
        print(f"  data.yaml not found at {yaml_path}")
        return None

    print("\n  Running official YOLO validation on test set...")
    metrics = model.val(data=str(yaml_path), split='test', augment=True)

    t = metrics.speed
    fps = 1000 / (t['preprocess'] + t['inference'] + t['postprocess'])

    print(f"  Official Test Recall:    {metrics.box.r[0]:.4f}")
    print(f"  Official Test Precision: {metrics.box.p[0]:.4f}")
    print(f"  Official Test mAP50:     {metrics.box.map50:.4f}")
    print(f"  Official Test FPS:       {fps:.2f}")

    return {
        'recall': float(metrics.box.r[0]),
        'precision': float(metrics.box.p[0]),
        'mAP50': float(metrics.box.map50),
        'fps': fps
    }


def test_detection(args, device):
    """
    Main YOLO detection test/evaluation pipeline.

    Runs:
    1. YOLO inference on processed test images
    2. (Optional) SAM refinement with IoU comparison
    3. Visual evidence generation
    4. Official YOLO validation metrics

    Args:
        args: Parsed arguments containing:
            - detection_model_type: RGB, REPLACEMENT, HSV_WAVELET, DIRECTIONAL, FUSION
            - yolo_dataset_path: Path to original Roboflow YOLO dataset
            - weights: Path to trained YOLO best.pt
            - use_sam: Whether to run SAM refinement
        device: torch device
    """
    model_type = args.detection_model_type.upper()
    print("=" * 50)
    print(f"  YOLO Detection Evaluation")
    print(f"  Model Type: {model_type}")
    print(f"  Device: {device}")
    print("=" * 50)

    # Load YOLO model
    weights_path = args.weights
    if not weights_path:
        weights_path = f'saved_models/best_yolo_{model_type.lower()}.pt'

    if not os.path.exists(weights_path):
        print(f"\n  ERROR: Weights not found at {weights_path}")
        print(f"  Train first with: python main.py --action train --task detection "
              f"--detection_model_type {model_type}")
        return

    print(f"\n  Loading YOLO model from: {weights_path}")
    yolo_model = YOLO(weights_path)

    # Dataset paths
    original_dataset = Path(args.yolo_dataset_path)
    processed_dir = Path(f'./processed_datasets/dataset_{model_type.lower()}')

    # Build processed dataset if it doesn't exist
    if not processed_dir.exists():
        print("  Processed dataset not found, building...")
        from engine.train_detection import build_processed_dataset
        build_processed_dataset(original_dataset, processed_dir, model_type)

    # Paths
    test_images_dir = processed_dir / 'test' / 'images'
    test_labels_dir = processed_dir / 'test' / 'labels'
    test_files = list(test_images_dir.glob('*.*'))

    if not test_files:
        print(f"  ERROR: No test images found in {test_images_dir}")
        return

    print(f"  Found {len(test_files)} test images")

    # Output directory for visual evidence
    visual_dir = Path(f'./runs/detection/{model_type.lower()}_eval/visual_evidence')
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM if requested
    use_sam = getattr(args, 'use_sam', False)
    sam_predictor = None
    if use_sam:
        sam_predictor = load_sam_model(device)
        if sam_predictor is None:
            print("  Continuing without SAM refinement...")
            use_sam = False

    # ==========================================
    # Evaluation Loop
    # ==========================================
    stats = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    ious_yolo = []
    ious_sam = []
    shrinkage_ratios = []
    saved_count = 0
    MAX_SAVE = 50

    for img_file in tqdm(test_files, desc="  Evaluating"):
        # Load processed image for YOLO
        processed_image = cv2.imread(str(img_file))
        if processed_image is None:
            continue
        h, w, _ = processed_image.shape

        # Load original RGB image (for SAM and visualization)
        orig_img_path = original_dataset / 'test' / 'images' / img_file.name
        if orig_img_path.exists():
            orig_image = cv2.imread(str(orig_img_path))
            orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        else:
            orig_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Ground truth
        label_file = test_labels_dir / f"{img_file.stem}.txt"
        gt_boxes = load_ground_truth_boxes(label_file, w, h)
        has_gt = len(gt_boxes) > 0

        # YOLO inference
        results = yolo_model(processed_image, verbose=False)
        found_rip = len(results[0].boxes) > 0

        # Update statistics
        if has_gt and found_rip:
            stats['TP'] += 1
        elif not has_gt and found_rip:
            stats['FP'] += 1
        elif has_gt and not found_rip:
            stats['FN'] += 1
        else:
            stats['TN'] += 1

        if not found_rip or not has_gt:
            continue

        # Best YOLO box vs first GT box
        yolo_box = results[0].boxes.xyxy.cpu().numpy()[0]
        gt_box = gt_boxes[0]

        iou_yolo = calculate_iou(yolo_box, gt_box)
        ious_yolo.append(iou_yolo)

        # SAM refinement
        iou_sam_val = iou_yolo
        refined_box = None
        sam_mask = None

        if use_sam and sam_predictor is not None:
            sam_predictor.set_image(orig_image_rgb)

            input_box = torch.tensor(yolo_box, device=sam_predictor.device)[None, :]
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                input_box, orig_image_rgb.shape[:2]
            )

            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

            sam_mask = masks[0, 0].cpu().numpy()
            refined_box = get_box_from_mask(sam_mask)

            if refined_box is not None:
                iou_sam_val = calculate_iou(refined_box, gt_box)

                # Shrinkage ratio
                area_yolo = (yolo_box[2] - yolo_box[0]) * (yolo_box[3] - yolo_box[1])
                area_sam = (refined_box[2] - refined_box[0]) * (refined_box[3] - refined_box[1])
                shrinkage_ratios.append(area_sam / (area_yolo + 1e-6))

        ious_sam.append(iou_sam_val)

        # Save visualization
        if saved_count < MAX_SAVE:
            save_visualization(
                orig_rgb=orig_image_rgb,
                gt_box=gt_box,
                yolo_box=yolo_box,
                refined_box=refined_box if use_sam else None,
                sam_mask=sam_mask,
                iou_yolo=iou_yolo,
                iou_sam=iou_sam_val,
                save_path=visual_dir / f"viz_{img_file.name}"
            )
            saved_count += 1

    # ==========================================
    # Results Summary
    # ==========================================
    print("\n" + "=" * 50)
    print(f"  Results: {model_type} ({len(test_files)} images)")
    print("-" * 50)

    total = stats['TP'] + stats['FP'] + stats['FN'] + stats['TN']
    if (stats['TP'] + stats['FN']) > 0:
        recall = stats['TP'] / (stats['TP'] + stats['FN'])
        precision = stats['TP'] / (stats['TP'] + stats['FP'] + 1e-6)
        print(f"  TP: {stats['TP']}  FP: {stats['FP']}  FN: {stats['FN']}  TN: {stats['TN']}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  Precision: {precision:.2%}")
    else:
        print("  No ground truth rip currents in test set.")

    if len(ious_yolo) > 0:
        print(f"\n  YOLO IoU (mean):     {np.mean(ious_yolo):.4f}")
        if use_sam and len(ious_sam) > 0:
            print(f"  YOLO+SAM IoU (mean): {np.mean(ious_sam):.4f}")
            improvement = np.mean(ious_sam) - np.mean(ious_yolo)
            print(f"  SAM Improvement:     {improvement:+.4f}")
        if len(shrinkage_ratios) > 0:
            print(f"  Avg Box Shrinkage:   {np.mean(shrinkage_ratios):.1%}")
    else:
        print("  No matching detections to calculate IoU.")

    print(f"\n  Visual evidence saved to: {visual_dir} ({saved_count} images)")

    # Official YOLO validation
    official_metrics = run_yolo_validation(yolo_model, processed_dir)

    print("=" * 50)

    return {
        'stats': stats,
        'ious_yolo': ious_yolo,
        'ious_sam': ious_sam,
        'shrinkage_ratios': shrinkage_ratios,
        'official_metrics': official_metrics
    }