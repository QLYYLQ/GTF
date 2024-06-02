# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))


from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
from torchvision import transforms


class VIFBdataset(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=(256, 256),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    ):
        self.root_dir = root_dir
        self.dir_name = ["IR", "VI"]
        self.visible_dir = os.path.join(root_dir, self.dir_name[1])
        self.infrared_dir = os.path.join(root_dir, self.dir_name[0])
        self.visible_list = self._list_all_files(self.visible_dir)
        self.infrared_list = self._list_all_files(self.infrared_dir)
        self.img_size = img_size
        self.transform = transform

    def _list_all_files(self, dir):
        path = Path(dir)
        image_list = []
        for filepath in path.iterdir():
            if filepath.is_file():
                image_list.append(str(filepath))
        return image_list

    def _change_image_to_tensor(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = cv2.resize(img, self.img_size, cv2.INTER_AREA)
        if img is not None:
            img_tensor = self.transform(img)
            return img_tensor

    def __len__(self):
        return len(self.visible_list)

    def __getitem__(self, index):
        vis_img = self._change_image_to_tensor(self.visible_list[index])
        inf_img = self._change_image_to_tensor(self.infrared_list[index])
        img_name = self.infrared_list[index].split("/")[-1]
        return vis_img, inf_img,img_name
