import cv2
import random
import torch.nn as nn
import torch
import os.path
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import Grayscale
from torchvision import transforms
from torch.utils import data


class DatasetCHUAC(data.Dataset):

    # datapath
    def __init__(self, benchmark, datapath, split, img_mode, img_size):
        super(DatasetCHUAC, self).__init__()
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.benchmark = benchmark
        assert self.benchmark == 'chuac'
        self.img_mode = img_mode
        assert img_mode in ['crop', 'same', 'resize']
        self.img_size = img_size
        print("self.img_size",self.img_size,self.split)
        self.img_path = os.path.join(datapath,'Original/')
        self.ann_path = os.path.join(datapath,'Photoshop/')
        self.img_metadata = self.load_metadata()

        self.norm_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449],  
                                 [0.226]) 
        ])

        if self.img_mode == 'resize' and self.split == 'train':
            self.resize = transforms.Resize([self.img_size, self.img_size], interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, index):
        img_name = self.img_metadata[index]
        img, anno_mask, org_img_size = self.load_frame(img_name)

        if self.split == 'train':
            img, anno_mask = self.augmentation(img, anno_mask)
        if self.img_mode == 'resize' and self.split == 'train':
            img = self.resize(img)
            anno_mask = self.resize(anno_mask)
        if self.img_mode == 'crop' and self.split == 'train':
            # 512to256
            i, j, h, w = self.get_params(img, (self.img_size, self.img_size))
            img = F.crop(img, i, j, h, w)
            anno_mask = F.crop(anno_mask, i, j, h, w)
        if self.img_mode == 'same' and self.split == 'test' or self.split == 'val':
            img = self.get_square(img)
            img = self.tensor_to_pil(img)
        img = self.norm_img(img)

        batch = {
            'img_name': img_name,
            'img': img,
            'anno_mask': anno_mask
        }
        return batch

    def tensor_to_pil(self, img):
        if len(img.size()) > 2:
            img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  
        else:
            img = img.cpu().numpy().astype(np.uint8)  
        return Image.fromarray(img)

    def get_square(self, img):
        if isinstance(img, Image.Image):
            img =np.array(img)
            img = torch.from_numpy(img)
            if len(img.size()) > 2:
                img = img.permute(2, 0, 1).float()  
            else:
                img = img.float()
        shape = 512

        if len(img.size()) > 2:
            _, h, w, = img.shape  
            pad = nn.ConstantPad2d((0, shape - w, 0, shape - h), 0)  
        else:
            h, w, = img.shape  # array list
            pad = nn.ConstantPad2d((0, shape - w, 0, shape - h), 0)  
        img = pad(img)
        return img

    def augmentation(self, img, anno_mask):

        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)
        anno_mask = transform_hflip(anno_mask)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)
        anno_mask = transform_vflip(anno_mask)

        if np.random.random() > 0.5:
            p = np.random.uniform(90, 270, 1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)
            anno_mask = transform_rotate(anno_mask)

        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[1.0, 2.1], contrast=[1.0, 2.1], saturation=[0.5, 1.5])
            img = color_aug(img)

        return img, anno_mask

    def load_frame(self, img_name):
        img_name = img_name.split()[0].split(".")[0]
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        org_img_size = img.size

        return img, anno_mask, org_img_size

    def read_mask(self, img_name):
        if int(img_name) <= 17 and int(img_name) >= 11:
            suffix = 'PNG'
        else:
            suffix = 'png'
        mask = np.array(Image.open(os.path.join(self.ann_path, 'angio'+img_name) + 'ok.'+suffix))
        mask[mask == 0] = 0
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_img(self, img_name):
        img = Image.open(os.path.join(self.img_path, img_name) + '.png')
        img = Grayscale(1)(img)
        img = np.array(img)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(img)

    def load_metadata(self):
        if self.split == 'train':
            meta_file = os.path.join('/data/split', self.benchmark, 'train.txt')
        elif self.split == 'val' or self.split == 'test':
            meta_file = os.path.join('/data/split', self.benchmark, 'test.txt')
        else:
            raise RuntimeError('Undefined split ', self.split)

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname # id

    def get_params(self, img, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw