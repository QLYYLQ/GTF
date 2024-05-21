import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from pytorch_msssim import MS_SSIM,ssim
import random
from dataloader import get_dataset,get_datapath
from model.net import NestFuse_light2_nodense,Fusion_network
from tqdm import tqdm,trange
import os
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
import cv2

nest_model = NestFuse_light2_nodense()
fusion_model = Fusion_network()