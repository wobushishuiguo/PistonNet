import os
import sys
import datetime
import pprint
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, str_gpus, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
from urllib.request import urlretrieve

def get_bboxes(cam, cam_thr=0.6):
    cam = np.array(cam)
    cam = (cam/cam.max() * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)


config_file = '../configs/CUB/deit_tscam_tiny_patch16_224.yaml'
cfg_from_file(config_file)
cfg.BASIC.ROOT_DIR = '../'
_ , val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=False,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
        )
model = model.cuda()
checkpoint = torch.load('../checkpoints/model_nw.pth')
pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(pretrained_dict)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
im_web = Image.open("../demo/hrsc/295.jpg").convert('RGB')
x_web = transform(im_web)


with torch.no_grad():
    x_web_logits, ts_cam_img_web = model(x_web.unsqueeze(0).cuda(), True)
pred_cls_id_web = x_web_logits.softmax(dim=-1)
print(pred_cls_id_web)
# pred_single_point = ts_cam_img_web[0]/ts_cam_img_web.max()
# target = torch.tensor([1])
# prec1, prec5 = accuracy(x_web_logits.data.contiguous().detach().cpu(), target, topk=(1,2))
# pred_c = x_web_logits.softmax(dim=1)
# print(pred_c)
mask_pred_web = ts_cam_img_web.detach().cpu().numpy()#pred_cls_id_web+1

v_min_web, v_max_web = mask_pred_web.min(), mask_pred_web.max()
mask_pred_web = (mask_pred_web - v_min_web) / (v_max_web - v_min_web)
mask_pred_web = mask_pred_web[1:].reshape([14,14])
mask_pred_web = cv2.resize(mask_pred_web, im_web.size)
# kkkkk = np.sign((2*mask_pred_web-1))
# mask_pred_web = 0.5+0.5*kkkkk*np.abs((2*mask_pred_web-1)) ** (1/3)
#plt.imshow(mask_pred_web)

result_web = (mask_pred_web[...,np.newaxis] * im_web).astype("uint8")

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
# plt.imshow(mask_pred_web)
# plt.show()
ax1.set_title('Original')
ax2.set_title('Attention Map')
ax3.set_title('mask')
_ = ax1.imshow(im_web)
_ = ax2.imshow(result_web)
_ = ax3.imshow(mask_pred_web)
plt.show()