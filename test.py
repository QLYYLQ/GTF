import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from torch.utils.data import DataLoader
from model.net import NestFuse_light2_nodense,Fusion_network
from tqdm import tqdm
import os
import torch
from dataloader import get_dataset,get_testdatapath,get_imgsize
import numpy as np
import cv2
from utils import test as tt

nest_path = [r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:1--gamma:70--w1:6--w2:3--ssim vi:1.0--ssim ir:0.5/nest model best.pth",r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:7--gamma:70--w1:6--w2:3--ssim vi:1.0--ssim ir:0.5/nest model best.pth"]# [r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:20--gamma:700w1:0.7--w2:0.3/nest model best.pth",r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:20--gamma:700w1:0.8--w2:0.2/nest model best.pth",r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:25--gamma:700w1:0.7--w2:0.3/nest model best.pth",r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/beta:25--gamma:700w1:0.8--w2:0.2/nest model best.pth"]
fusion_path = [i.replace("nest","fusion") for i in nest_path]
saved_path = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved/test"
def test():
    for i in range(4):
        nest_model = NestFuse_light2_nodense([64,112,160,208],1,1,False)
        fusion_model = Fusion_network([64,112,160,208],"res")
        nest = nest_path[i]
        fusion = fusion_path[i]
        nest_model.load_state_dict(torch.load(nest)["model"])
        fusion_model.load_state_dict(torch.load(fusion)["model"])
        nest_model.cuda().eval()
        fusion_model.cuda().eval()
        nest = nest.split("/")[-2]
        dataset_name = ["RoadScene","VIFB"]
        for i in dataset_name:
            path = os.path.join(saved_path,i,nest)
            os.makedirs(path,exist_ok=True)
            train(nest_model,fusion_model,i,path)


def train(nest_model,fusion_model,dataset_name,path):
    dataset = get_dataset(dataset_name)(get_testdatapath(dataset_name))
    img_size = get_imgsize(dataset_name)
    loader = DataLoader(dataset,batch_size=6,drop_last=True)
    load = tqdm(loader,total=len(loader))
    ag = []
    mi = []
    en =[]
    ssim = []
    sf = []
    for batch in load:
        img_vis,img_inf,file_name = batch
        img_vis = img_vis.cuda()
        img_inf = img_inf.cuda()[:, 0, :, :].unsqueeze(1)
        img_vis1 = img_vis.cuda()[:, 0, :, :].unsqueeze(1)
        with torch.no_grad():
            en_ir = nest_model.encoder(img_inf)
            en_vi = nest_model.encoder(img_vis1)
            f = fusion_model(en_ir,en_vi)
            outputs = nest_model.decoder_eval(f)
            for output in outputs:
                img = 0.69*img_vis1+0.31*output
                for i in range(img.shape[0]):
                    a = img[i].detach().cpu().squeeze().numpy()
                    b = img_vis1[i].detach().cpu().squeeze().numpy()
                    ag.append(tt.avgGradient(a))
                    mi.append(tt.getMI(a,b))
                    sf.append(tt.spatialF(a))
                    en.append(tt.getEn(a))
                    # if output.shape[0]!=6:
                    #     print(1)
                    name = file_name[i]
                    path1 = os.path.join(path,name)
                    save_img_as_ycbcr(img[i],path1,img_size)
    print()
    print(sum(mi)/len(mi))
    print(sum(ag)/len(ag))
    print(sum(sf)/len(sf))
    print(sum(en)/len(en))


def save_img_as_ycbcr(img:torch.Tensor,path:str,img_size):

    img = img*255
    img = cv2.resize(img.detach().squeeze().cpu().numpy(),img_size)
    cv2.imwrite(path,img)
        

if __name__ == "__main__":
    test()
