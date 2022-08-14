import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from common.utils.pose_utils import process_bbox
from data.dataset import generate_patch_image
from main.config import cfg
from main.model import get_root_net


if __name__ == "__main__":
    model_path = "demo/snapshot_18.pth.tar"
    model = get_root_net(cfg, False)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()
