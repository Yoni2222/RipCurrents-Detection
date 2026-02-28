Installation & Requirements
It is highly recommended to use Python 3.10 or 3.11 to ensure full compatibility with PyTorch and CUDA.

1. Create and activate a virtual environment:

2. Install PyTorch with GPU (CUDA) support:

3. Install project dependencies:

Usage (CLI)
The project is fully controlled via the Command Line Interface (CLI). There is no need to modify the code to switch models or datasets.

Mandatory Arguments:
--action: What action to perform (train or test).

--task: The type of task (classification or detection).

--model_type: The specific model architecture to run (see lists below).

1. Classification
Supported Models: rgb_baseline, single_rgb_wavelet, single_wavelet_only, channel_replacement, dual_stream_4_3, dual_stream_4_2, dual_stream_3_2, dual_stream_3_3.

Example: Train a Dual Stream model (4 Wavelet channels + 3 RGB channels):
"python main.py --action train --task classification --model_type dual_stream_4_3"

Example: Test a model (requires a path to the saved weights):
"python main.py --action test --task classification --model_type dual_stream_4_3 --weights_path path_to_weights.pth"

2. Object Detection (YOLO + SAM)
Supported Models: yolo_rgb, yolo_hsv_wavelet, yolo_directional, yolo_fusion, yolo_replacement.

Example: Train YOLO with HSV + Wavelet transformed images:
(Note: The system will automatically generate a dedicated dataset folder named dataset_yolo_hsv_wavelet from the raw source directory).
"python main.py --action train --task detection --model_type yolo_hsv_wavelet"

Example: Test YOLO + SAM Hybrid approach:
(Note: During the first run, the SAM model weights - approx. 2.4GB - will be downloaded automatically).
"python main.py --action test --task detection --model_type yolo_hsv_wavelet --weights_path path_to_yolo_weights.pth"

Upon completing the test phase for detection models, a Visual_Evidence folder will be generated alongside the weights. This folder contains images displaying the YOLO bounding box (Red), the refined SAM bounding box/mask (Cyan), and the Ground Truth (Green).

Reproducibility
To ensure academic and scientific integrity, this project includes strict settings to maintain deterministic behavior across runs:

Environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 is enforced.

PyTorch is strictly configured to use deterministic algorithms on the GPU (torch.use_deterministic_algorithms(True)).
