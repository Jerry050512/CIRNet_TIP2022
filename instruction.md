# CIRNet Training and Testing Instructions

This document provides instructions on how to set up the environment and run the training and testing scripts for the CIRNet model on the NEU-RSDDS-AUG dataset.

## 1. Environment Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (for training and testing)

### Dependencies
Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## 2. Dataset Configuration

The model is configured to use the **NEU-RSDDS-AUG** dataset. Ensure your dataset is structured as follows:

```
../datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/      # .bmp files
    ├── Depth_train/      # .tiff files
    ├── GT_train/         # .png files
    ├── Image_test/       # .bmp files
    └── Depth_test/       # .tiff files
```

The scripts expect this directory structure relative to the project root.

## 3. Training

To start the training process, run the `CIRNet_train.py` script from the project's root directory:

```bash
python3 CIRNet_train.py
```

### Training Outputs
- **Log File**: Training progress is logged to `/hy-tmp/output/result.log`.
- **Checkpoints**: The model checkpoint is saved and overwritten at `/hy-tmp/output/checkpoint.pth`. Checkpoints are saved periodically (every 5 epochs after epoch 60) and at the end of training.

## 4. Testing (Prediction)

After training, a checkpoint file will be available at `/hy-tmp/output/checkpoint.pth`. To run predictions on the test set, execute the `CIRNet_test.py` script:

```bash
python3 CIRNet_test.py
```

### Testing Outputs
- **Log File**: Testing logs are appended to `/hy-tmp/output/result.log`.
- **Predictions**: The output prediction maps (as PNG images) are saved to `/hy-tmp/output/predictions/`. The predictions are resized to match the original input image dimensions.

## 5. Modifying Hyperparameters

All default configurations, including data paths, learning rate, batch size, and backbone model, are set in `options.py`. You can modify this file to change the default behavior or pass arguments via the command line. For example, to train for a different number of epochs:
```bash
python3 CIRNet_train.py --epoch 200
```