import os
import time
import random
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from options import opt
from datetime import datetime

from model.CIRNet_Res50 import CIRNet_R50
from model.CIRNet_vgg16 import CIRNet_V16

from utils import clip_gradient, adjust_lr
from dataLoader import get_loader

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

seed_torch()

# set the path
image_root = opt.rgb_root
depth_root = opt.depth_root
gt_root = opt.gt_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# load data
print('load data...')
train_loader = get_loader(image_root, depth_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=os.path.join(save_path, 'result.log'), format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CIRNet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.
             format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
                    opt.decay_epoch))

# load model
print('load model...')
if opt.backbone == 'R50':
    model = CIRNet_R50()
else:
    model = CIRNet_V16()
print('Use backbone: ' + opt.backbone)

# load gpu
if torch.cuda.is_available():
    model.cuda()
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        print("Use multiple GPUs -", opt.gpu_id)
        model = torch.nn.DataParallel(model)

# Restore training from checkpoints
if opt.load is not None and os.path.exists(opt.load):
    model.load_state_dict(torch.load(opt.load))
    print('load model from', opt.load)

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
CE = torch.nn.BCEWithLogitsLoss()

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, depths, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                images = images.cuda()
                depths = depths.cuda()
                gts = gts.cuda()

            s_rgb, s_depth, s_rgbd = model(images, depths)
            loss_r = CE(s_rgb, gts)
            loss_d = CE(s_depth, gts)
            loss_rd = CE(s_rgbd, gts)
            loss = loss_r + loss_d + loss_rd

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            epoch_step += 1
            loss_all += loss_rd.item() # Use .item() to get scalar value

            if i % 50 == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.6f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))

        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}]:Loss_AVG={:.6f}'.format(epoch, opt.epoch, loss_all))
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        if epoch > 0 and (epoch % 5 == 0 or epoch == opt.epoch):
            torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))
            print('Checkpoint saved to {}'.format(os.path.join(save_path, 'checkpoint.pth')))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))
        print('save checkpoint successfully!')
        raise

if __name__ == '__main__':
    print("Start train...")
    time_begin = time.time()
    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # No SummaryWriter needed
        train(train_loader, model, optimizer, epoch, save_path)
        time_epoch = time.time()
        print("Time out:{:.2f}s\n".format(time_epoch - time_begin))