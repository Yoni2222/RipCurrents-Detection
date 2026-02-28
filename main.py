import torch
import os
import random
import numpy as np
from args_parser import parse_args
from engine.train_classification import train_classification
from engine.test_classification import test_classification

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"\n SEED SET TO: {seed}")

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting {args.action.upper()} for {args.task} using {args.model_type.upper()}")
    print(f"Device: {device}")

    if args.task == 'classification':
        if args.action == 'train':
            print(f"Training parameters: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}")
            train_classification(args, device)

        elif args.action == 'test':
            if not args.weights:
                raise ValueError("Must provide --weights path for testing!")
            print(f"Testing with weights: {args.weights}")
            test_classification(args, device)

    elif args.task == 'detection':
        if args.action == 'train':
            from engine.train_detection import train_detection
            print(f"Training parameters: Epochs={args.epochs}, Batch={args.batch_size}, "
                f"YOLO size={args.yolo_model_size}")
            train_detection(args, device)
        elif args.action == 'test':
            from engine.test_detection import test_detection
            if not args.weights:
                default_weights = f'saved_models/best_yolo_{args.detection_model_type.lower()}.pt'
                if os.path.exists(default_weights):
                    args.weights = default_weights
                    print(f"Using default weights: {args.weights}")
                else:
                    raise ValueError("Must provide --weights path for testing! "
                             f"(or train first to generate {default_weights})")
            print(f"Testing with weights: {args.weights}")
            if args.use_sam:
                print("SAM refinement: ENABLED")

            test_detection(args, device)

if __name__ == "__main__":
    main()