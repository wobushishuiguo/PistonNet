import os
import glob
from tqdm import tqdm
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

def dis(x1,y1,x2,y2):
    distance = np.sqrt(((x2-x1)**2+(y2-y1)**2))
    return distance
def get_bboxes(cam, cam_thr=0.6):
    cam = np.array(cam)[:,:,0]
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

    return contours#estimated_bbox  #, thr_gray_heatmap, len(contours)

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
checkpoint = torch.load('../checkpoints/model_epoch60.pth')
pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(pretrained_dict)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

path = '../val/test'
img_list = glob.glob(path +'/*.png')
dx = (0,208,288)
dy = (0,208,288)
recall_sum=0
precision_sum=0
target_sum =0
pre_sum=0
for id, img_file in tqdm(enumerate(img_list)):
    im_web = Image.open(img_file).convert('RGB')
    final_image = Image.new('RGB', (512, 512), (0, 0, 0))
    for i in range(3):
        for j in range(3):
            new_image = im_web.crop((dx[i], dy[j], dx[i] + 224, dy[j] + 224))
            x_web = transform(new_image)
            x_web_logits,  cams = model(x_web.unsqueeze(0).cuda(), True)
            pred_cls_id_web = x_web_logits.softmax(dim=1)
            pred_single_point = 1-cams[0]
            pred_total = pred_cls_id_web[0,1]*pred_single_point+pred_cls_id_web[0,1]*(1-pred_cls_id_web[0,1])
            if pred_single_point > 0.95:
                cams=cams.detach().cpu().numpy()
                #cams = cams[1:].reshape([14,14])
                v_min_cam, v_max_cam = cams.min(), cams.max()
                mask_cam = (255 * (cams - v_min_cam) / (v_max_cam - v_min_cam))
                mask_cam = mask_cam[1:].reshape([14,14])
                mask_cam = cv2.resize(mask_cam, (224, 224)).astype(np.int)
                mask_cam = Image.fromarray(mask_cam.astype('uint8')).convert('RGB')
                final_image.paste(mask_cam, (dx[i], dy[j], dx[i] + 224, dy[j] + 224))

    txt_file = img_file.replace('png', 'txt')
    txt_file = txt_file.replace('test', 'labeltxt')
    bboxes_gt = np.loadtxt(txt_file, skiprows=2, usecols=(0, 1, 2, 5))
    targets = len(bboxes_gt.shape)
    if targets == 1:
        bboxes_gt = bboxes_gt[np.newaxis, :]
    col = bboxes_gt.shape[0]
    bboxes_pred = get_bboxes(final_image, cam_thr=0.1)
    row = len(bboxes_pred)
    count = np.zeros((row,col),dtype='uint8')
    if row ==0:
        recall = 0
        pre_sum=pre_sum+row
        target_sum=target_sum+col
        precision=0
        recall_sum =recall_sum+recall
        precision_sum=precision_sum+precision
        continue
    for i,bbox_single in enumerate(bboxes_pred):
        center_x = (bbox_single[:, :, 0].min() + bbox_single[:, :, 0].max()) / 2
        center_y = (bbox_single[:, :, 1].min() + bbox_single[:, :, 1].max()) / 2
        for j,bbox_gt in enumerate(bboxes_gt):
            cx = (bbox_gt[ 0] + bbox_gt[ 2]) / 2
            cy = (bbox_gt[ 1] + bbox_gt[ 3]) / 2
            if center_x>=bbox_gt[0] and center_x<=bbox_gt[2] and center_y>=bbox_gt[1] and center_y<=bbox_gt[3]:
                count[i][j]=1
    #计算recall precision
    recall = count.max(axis = 0).sum()
    pre_sum = pre_sum+row
    target_sum = target_sum+col
    precision = count.max(axis = 1).sum()
    recall_sum=recall_sum+recall
    precision_sum=precision_sum+precision
print(recall_sum,target_sum,precision_sum,pre_sum)





