import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.classification_models import SingleStreamCNN, DualStreamRipDetectorWithAttention
from data_pipeline.dataset import get_dataset_class
from utils.metrics import calculate_full_metrics, find_optimal_threshold_by_acc


def get_predictions(model, loader, device, model_type):
    dual_stream_models = ['dual_stream_4_3', 'dual_stream_4_2', 'dual_stream_3_2', 'dual_stream_3_3']
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in loader:
            if model_type in dual_stream_models:
                rgb, wav, labels, _ = batch_data
                rgb, wav = rgb.to(device), wav.to(device)
                outputs = model(rgb, wav).squeeze()
            else:  # single stream
                imgs, labels = batch_data
                imgs = imgs.to(device)
                outputs = model(imgs).squeeze()

            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def test_classification(args, device):
    dual_stream_models = ['dual_stream_4_3', 'dual_stream_4_2', 'dual_stream_3_2', 'dual_stream_3_3']
    print(f" Starting Testing Phase for {args.model_type}...")

    if args.model_type in dual_stream_models:
        test_transform = None

    elif args.model_type == 'channel_replacement':
        test_transform = transforms.Compose([transforms.ToTensor()])

    elif args.model_type in ['rgb_baseline', 'single_rgb_wavelet', 'single_wavelet_only']:
        test_transform_list = [
            transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=int(img.height * 0.8),
                                                                     width=img.width)),
            transforms.Resize((320, 320)),
        ]
        if args.model_type == 'rgb_baseline':
            test_transform_list.append(transforms.ToTensor())

        test_transform = transforms.Compose(test_transform_list)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


    datasetClass = get_dataset_class(args.model_type)
    val_ds = datasetClass(args.dataset_path, 'valid', transform=test_transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    test_ds = datasetClass(args.dataset_path, 'test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)


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

    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f" Weights loaded from {args.weights}")
    except Exception as e:
        raise RuntimeError(f" Failed to load weights: {e}")

    val_probs, val_labels = get_predictions(model, val_loader, device, args.model_type)
    test_probs, test_labels = get_predictions(model, test_loader, device, args.model_type)

    acc_50, prec_50, rec_50, f1_50 = calculate_full_metrics(test_probs, test_labels, 0.5)

    print(f"\n RESULTS FOR {args.model_type.upper()}:")
    print(f"\n   Threshold = 0.5:")
    print(f"   Real Test Acc:        {acc_50:.2%}")
    print(f"   Real Test F1:         {f1_50:.2%}")
    print(f"   Real Test Precision:  {prec_50:.2%}")
    print(f"   Real Test Recall:     {rec_50:.2%}")