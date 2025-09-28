import os
import cv2
import time
import numpy as np
import logging

import torch
import torch.nn.functional as F
from PIL import Image

from options import opt
from model.CIRNet_Res50 import CIRNet_R50
from model.CIRNet_vgg16 import CIRNet_V16
from dataLoader import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# Configure logging
log_path = os.path.join(opt.save_path, 'result.log')
logging.basicConfig(filename=log_path, format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# load the model
print('load model...')
if opt.backbone == 'R50':
    model = CIRNet_R50()
else:
    model = CIRNet_V16()
print('Use backbone: ' + opt.backbone)

if torch.cuda.is_available():
    model.cuda()
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        print("Use multiple GPUs -", opt.gpu_id)
        model = torch.nn.DataParallel(model)

try:
    model.load_state_dict(torch.load(opt.test_model, map_location='cpu' if not torch.cuda.is_available() else None))
    print('Model loaded from {}'.format(opt.test_model))
except FileNotFoundError:
    print(f"Error: Model file not found at {opt.test_model}")
    exit()

model.eval()

# Define dataset paths
dataset_path = opt.test_path
image_root = os.path.join(dataset_path, 'Image_test/')
depth_root = os.path.join(dataset_path, 'Depth_test/')
# gt_root is not used for prediction, so it can be None
gt_root = None

# Define save path for predictions
save_path = os.path.join(opt.save_path, 'predictions')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)

print("Testing NEU-RSDDS-AUG...")
logging.info("Testing NEU-RSDDS-AUG...")

for i in range(test_loader.size):
    image, depth, _, name, original_size = test_loader.load_data()

    if torch.cuda.is_available():
        image = image.cuda()
        depth = depth.cuda()

    _, _, pre = model(image, depth)

    # Resize prediction to original image size
    pre_s = F.interpolate(pre, size=original_size[::-1], mode='bilinear', align_corners=False)

    pre_s = pre_s.sigmoid().data.cpu().numpy().squeeze()
    pre_s = (pre_s - pre_s.min()) / (pre_s.max() - pre_s.min() + 1e-8)

    # Save prediction as PNG
    cv2.imwrite(os.path.join(save_path, name), pre_s * 255)

print("Testing completed. Predictions saved to {}".format(save_path))
logging.info("Testing completed. Predictions saved to {}".format(save_path))
print("Test Done!")