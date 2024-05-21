import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))




from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
import numpy as np
import torch


class TNOdataset(Dataset):
    def __init__(self,root_dir,img_size = (256,256)):
        self.root_dir = root_dir
        self.dir_name = ["Inf","Vis"]
        self.visible_dir = os.path.join(root_dir,self.dir_name[1])
        self.infrared_dir = os.path.join(root_dir,self.dir_name[0])
        self.visible_list = self._list_all_files(self.visible_dir)
        self.infrared_list = self._list_all_files(self.infrared_dir)
        self.img_size = img_size
    def _list_all_files(self,dir):
        path = Path(dir)
        image_list = []
        for filepath in path.iterdir():
            if filepath.is_file():
                image_list.append(str(filepath))
        return image_list
    def _change_image_to_tensor(self,image_path):
        img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,self.img_size,cv2.INTER_AREA)
        if img is not None:
            img = img.astype(np.float32)/255.0
            img = np.expand_dims(img,axis=0)
            img_tensor = torch.from_numpy(img)
            return img_tensor
    def __len__(self):
        return len(self.visible_list)
    def __getitem__(self, index):
        vis_img = self._change_image_to_tensor(self.visible_list[index])
        inf_img = self._change_image_to_tensor(self.infrared_list[index])
        return vis_img,inf_img