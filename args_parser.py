import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RipCurrentDetection Pipeline")

    parser.add_argument('--action', type=str, choices=['train', 'test', 'predict'], required=True)
    parser.add_argument('--task', type=str, choices=['classification', 'detection'], default='classification')

    parser.add_argument('--model_type', type=str, default='dual_stream_4_3', choices=[
        'rgb_baseline',  # Single Stream RGB (3 channels)
        'single_rgb_wavelet',  # Single Stream (3 RGB + 1 Wavelet = 4 channels)
        'single_wavelet_only',  # Single Stream (4 Wavelet channels)
        'channel_replacement',  # Single Stream (Wavelet, G, B)
        'dual_stream_4_3',  # Dual Stream (4 Wavelet + 3 RGB)
        'dual_stream_4_2',  # Dual Stream (4 Wavelet + 2 RGB)
        'dual_stream_3_2',  # Dual Stream (3 Wavelet + 2 RGB)
        'dual_stream_3_3'  # Dual Stream (3 Wavelet + 3 RGB)
    ])

    parser.add_argument('--dataset_path', type=str, default='./rip_current_binary_dataset')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)

    # ==========================================
    # Detection-specific (YOLO)
    # ==========================================
    parser.add_argument('--detection_model_type', type=str, default='REPLACEMENT', choices=[
        'RGB',
        'REPLACEMENT',
        'HSV_WAVELET',
        'DIRECTIONAL',
        'FUSION'
    ], help='Detection preprocessing type for YOLO')

    parser.add_argument('--yolo_dataset_path', type=str, default='./Rip-Current-Monitoring-1',
                        help='Path to Roboflow YOLO dataset (with train/valid/test splits)')

    parser.add_argument('--yolo_model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)')

    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM refinement during evaluation (requires segment-anything)')


    return parser.parse_args()