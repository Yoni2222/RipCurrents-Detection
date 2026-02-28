import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import copy
import numpy as np
import random
from pathlib import Path

from models.classification_models import SingleStreamCNN, DualStreamRipDetectorWithAttention
from data_pipeline.dataset import get_dataset_class
from utils.metrics import calculate_metrics


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_classification(args, device):
    print(f"\n STARTING TRAIN LOOP FOR SEED: {args.seed}")

    if 'dual_stream' in args.model_type:
        train_transform = None
        val_transform = None

    elif args.model_type == 'channel_replacement':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif args.model_type in ['rgb_baseline', 'single_rgb_wavelet', 'single_wavelet_only']:
        train_transform_list = [
            transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=int(img.height * 0.8),
                                                                     width=img.width)),
            transforms.Resize((320, 320)),
            transforms.RandomHorizontalFlip(),
        ]
        val_transform_list = [
            transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=int(img.height * 0.8),
                                                                     width=img.width)),
            transforms.Resize((320, 320)),
        ]

        if args.model_type == 'rgb_baseline':
            train_transform_list.append(transforms.ToTensor())
            val_transform_list.append(transforms.ToTensor())

        train_transform = transforms.Compose(train_transform_list)
        val_transform = transforms.Compose(val_transform_list)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    datasetClass = get_dataset_class(args.model_type)
    train_ds = datasetClass(args.dataset_path, 'train', transform=train_transform)
    val_ds = datasetClass(args.dataset_path, 'valid', transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, worker_init_fn=seed_worker, generator=g)

    if args.model_type == 'rgb_baseline':
        model = SingleStreamCNN(input_channels=3).to(device)
    elif args.model_type == 'single_rgb_wavelet':
        model = SingleStreamCNN(input_channels=4).to(device)
    elif args.model_type == 'single_wavelet_only':
        model = SingleStreamCNN(input_channels=4).to(device)
    elif args.model_type == 'channel_replacement':
        model = SingleStreamCNN(input_channels=3).to(device)

    elif args.model_type == 'dual_stream_4_3':
        model = DualStreamRipDetectorWithAttention(rgb_in=3, wav_in=4).to(device)
    elif args.model_type == 'dual_stream_4_2':
        model = DualStreamRipDetectorWithAttention(rgb_in=2, wav_in=4).to(device)
    elif args.model_type == 'dual_stream_3_2':
        model = DualStreamRipDetectorWithAttention(rgb_in=2, wav_in=3).to(device)
    #elif args.model_type == 'dual_stream_3_3':
    #    model = DualStreamRipDetectorWithAttention(rgb_in=3, wav_in=3).to(device)
    else:
        model = DualStreamRipDetectorWithAttention(rgb_in=3, wav_in=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = model.state_dict()

    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        t_tp, t_fp, t_fn, t_correct, t_total = 0, 0, 0, 0, 0

        for batch_data in train_loader:
            optimizer.zero_grad()

            if 'dual_stream' in args.model_type:
                rgb, wav, labels, _ = batch_data
                rgb, wav, labels = rgb.to(device), wav.to(device), labels.to(device)
                outputs = model(rgb, wav).squeeze()
            else:  # All Single Stream variations
                # Some datasets return 2 items (img, label), some return 3 (img, label, filename)
                if len(batch_data) == 3:
                    imgs, labels, _ = batch_data
                else:
                    imgs, labels = batch_data

                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            t_correct += (preds == labels).sum().item()
            t_total += labels.size(0)
            t_tp += ((preds == 1) & (labels == 1)).sum().item()
            t_fp += ((preds == 1) & (labels == 0)).sum().item()
            t_fn += ((preds == 0) & (labels == 1)).sum().item()

        t_acc, t_prec, t_rec, t_f1 = calculate_metrics(t_tp, t_fp, t_fn, t_total, t_correct)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        v_tp, v_fp, v_fn, v_correct, v_total = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch_data in val_loader:
                if 'dual_stream' in args.model_type:
                    rgb, wav, labels, _ = batch_data
                    rgb, wav, labels = rgb.to(device), wav.to(device), labels.to(device)
                    out = model(rgb, wav).squeeze()
                else:
                    imgs, labels = batch_data
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs).squeeze()

                val_loss += criterion(out, labels).item()
                preds = (out > 0.5).float()
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                v_tp += ((preds == 1) & (labels == 1)).sum().item()
                v_fp += ((preds == 1) & (labels == 0)).sum().item()
                v_fn += ((preds == 0) & (labels == 1)).sum().item()

        v_acc, v_prec, v_rec, v_f1 = calculate_metrics(v_tp, v_fp, v_fn, v_total, v_correct)

        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  [Train] Loss: {train_loss / len(train_loader):.4f} | Acc: {t_acc:.2%} | F1: {t_f1:.2%}")
        print(f"  [Valid] Loss: {val_loss / len(val_loader):.4f} | Acc: {v_acc:.2%} | F1: {v_f1:.2%}")


        if v_f1 > best_val_f1:
            print(f"  NEW BEST MODEL! (Previous F1: {best_val_f1:.4f} -> New: {v_f1:.4f})")
            best_val_f1 = v_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            save_name = f"best_{args.model_type}.pth"
            torch.save(best_model_state, save_dir / save_name)
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            print(f"\n Early stopping triggered at epoch {epoch + 1}")
            break