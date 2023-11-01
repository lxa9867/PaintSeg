import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import time
import cv2
import json


class Base(Dataset):
    def __init__(self,
                 txt_file,
                 degradation,
                 state,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,

                 ):

        self.data_root = '/opt/project/src/additional_data/'
        print('began')

        with open('all_caption.json', 'r') as fp:
            data = json.load(fp)
        if state == 'val':
            self.image_paths = data[:1000]
        else:
            self.image_paths = data[1000:]
        self.size = size

        self.labels = {
            "relative_file_path_": [0 for l in self.image_paths],
            "file_path_": self.image_paths,
        }
        self._length = len(self.image_paths)
        print(f'state: {state}, dataset size:{self._length}')
        self.hr_height, self.hr_width = (256, 256)

    def __getitem__(self, i):
        example = {}
        image_path = self.data_root + self.image_paths[i]
        image = cv2.imread(image_path.replace('teeth_info.txt', 'im.jpg'))
        with open(image_path, 'r') as f:
            txt = f.read()
        text = txt.replace('\n', '\n ')
        image = cv2.resize(image, (256, 256))
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["caption"] = text
        return example

    def __len__(self):
        return self.size


class Txttrain(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", state='train', **kwargs)


class Txtval(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", state='val', **kwargs)