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

nest_model = NestFuse_light2_nodense([64, 112, 160, 208],1,1,False).cuda()
fusion_model = Fusion_network([64, 112, 160, 208],'res').cuda()
nest = torch.load("/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/600--w1:6.0--w2:3.0/nest model best.pth")
fusion = torch.load("/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/600--w1:6.0--w2:3.0/fusion model best.pth")
# nest = torch.load(r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/model/nestfuse_gray_1e2.model")

nest_model.load_state_dict(nest["model"])
fusion_model.load_state_dict(fusion["model"])

test_dataset_dir  = {"TNO":"/root/autodl-tmp/test_for_paper/Code_For_ITCD/dataset/test/TNO"}
eval_name = ["TNO"]
EPSILON=1e-6

def set_seed(seed=42):
    """为PyTorch、NumPy和Python内置的random库设置种子，以确保实验可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    # 确保PyTorch的卷积操作的确定性，这可能会降低一些运行效率
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)




def eval1(eval_name):
    for i in eval_name:
        eval_each_dataset(i,fusion_model,nest_model)


def eval_each_dataset(eval_name,fusion_model,nest_model):
    dataset = get_dataset(eval_name)(test_dataset_dir[eval_name])
    dataloader = DataLoader(dataset,batch_size=6,num_workers=3)
    with torch.no_grad():
        for index,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            img_vi,img_ir =batch
            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()
            en_vi = nest_model.encoder(img_vi)
            en_ir = nest_model.encoder(img_ir)
            f = fusion_model(en_ir,en_vi)
            out = nest_model.decoder_eval(f)
            output = out[0]
            output = output / (torch.max(output) - torch.min(output) + EPSILON)
            # out = out[0]
            save_img(output,r"/root/autodl-tmp/test_for_paper/Code_For_ITCD")


def save_img(img,path):
	batch = img.shape[0]
	for i in range(batch):
		img = img.detach()
		img_numpy = img[i].squeeze().cpu().numpy()*255
		img_resize = cv2.resize(img_numpy,(640,512),cv2.INTER_AREA)
		path1 = os.path.join(path,f"第{i}张图.png")
		cv2.imwrite(str(path1),img_resize)

if __name__ == "__main__":
    eval1(["TNO"])

