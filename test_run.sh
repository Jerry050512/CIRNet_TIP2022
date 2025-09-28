#!/bin/bash

# Exit on any error
set -e

echo "Creating local dummy dataset and output directories..."

# Create base directories locally
mkdir -p ./datasets/NEU-RSDDS-AUG/Image_train
mkdir -p ./datasets/NEU-RSDDS-AUG/Depth_train
mkdir -p ./datasets/NEU-RSDDS-AUG/GT_train
mkdir -p ./datasets/NEU-RSDDS-AUG/Image_test
mkdir -p ./datasets/NEU-RSDDS-AUG/Depth_test
mkdir -p ./output/predictions

echo "Generating dummy image files..."

# Create dummy training files
python3 -c "from PIL import Image; import numpy as np; \
img = Image.new('RGB', (352, 352), color = 'red'); \
img.save('./datasets/NEU-RSDDS-AUG/Image_train/train_1.bmp'); \
depth = Image.fromarray(np.zeros((352,352), dtype=np.uint16)); \
depth.save('./datasets/NEU-RSDDS-AUG/Depth_train/train_1.tiff'); \
gt = Image.new('L', (352, 352), color = 255); \
gt.save('./datasets/NEU-RSDDS-AUG/GT_train/train_1.png')"

# Create dummy testing files
python3 -c "from PIL import Image; import numpy as np; \
img = Image.new('RGB', (352, 352), color = 'blue'); \
img.save('./datasets/NEU-RSDDS-AUG/Image_test/test_1.bmp'); \
depth = Image.fromarray(np.zeros((352,352), dtype=np.uint16)); \
depth.save('./datasets/NEU-RSDDS-AUG/Depth_test/test_1.tiff')"

echo "Dummy data created."

# Backup original files
cp options.py options.py.bak
cp CIRNet_train.py CIRNet_train.py.bak

echo "Temporarily modifying scripts for local test run..."
# Change paths to local
sed -i "s|'../datasets/NEU-RSDDS-AUG/|'./datasets/NEU-RSDDS-AUG/|g" options.py
sed -i "s|'/hy-tmp/output/|'./output/|g" options.py
# Change checkpoint condition to trigger on any epoch
sed -i "s/if epoch > 60/if epoch > 0/" CIRNet_train.py


echo "Running training script (1 epoch)..."
python3 CIRNet_train.py --epoch 1 --batchsize 1

echo "Training script finished."

echo "Running testing script..."
python3 CIRNet_test.py

echo "Testing script finished."

# Restore original files
mv options.py.bak options.py
mv CIRNet_train.py.bak CIRNet_train.py
echo "Restored original scripts."

echo "Verifying outputs..."
if [ -f ./output/checkpoint.pth ]; then
    echo "OK: Checkpoint file found."
else
    echo "FAIL: Checkpoint file not found."
    exit 1
fi

if [ -f ./output/result.log ]; then
    echo "OK: Log file found."
else
    echo "FAIL: Log file not found."
    exit 1
fi

if [ -f ./output/predictions/test_1.png ]; then
    echo "OK: Prediction file found."
else
    echo "FAIL: Prediction file not found."
    exit 1
fi

# Clean up local directories
rm -rf ./datasets
rm -rf ./output

echo "Test run completed and cleaned up successfully!"