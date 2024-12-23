import os.path
import torch
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from einops import rearrange


class Iharmony4Dataset(Dataset):
    def __init__(self, dataset_root, is_for_train=True, resize=None):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super().__init__()
        self.root = dataset_root
        self.isTrain = is_for_train
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self._load_images_paths()
        self.transform = HarmonyTransform(resize=(resize,resize))

    def _load_images_paths(self,):
        if self.isTrain:
            trainfile = os.path.join(self.root, 'IHD_train.txt')
            with open(trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    # data_parts = line.split('/')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    
                    self.image_paths.append(os.path.join(self.root, line))
                    self.mask_paths.append(os.path.join(self.root, mask_path))
                    self.gt_paths.append(os.path.join(self.root, gt_path))
        else:
            # if self.opt.accelerator.is_main_process:
            trainfile = os.path.join(self.root, 'IHD_test.txt')
            with open(trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    
                    self.image_paths.append(os.path.join(self.root, line))
                    self.mask_paths.append(os.path.join(self.root, mask_path))
                    self.gt_paths.append(os.path.join(self.root, gt_path))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert('RGB')
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('1')
        
        #apply the same transform to composite and real images
        comp, real, mask = self.transform(comp, real, mask)
        
        return {'comp': comp, 'mask': mask, 'real': real, 'img_path':self.image_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


class HarmonyTransform():
    def __init__(
        self, 
        mean_img=[0.5, 0.5, 0.5],
        mean_mask=[0.5],
        std_img=[0.5, 0.5, 0.5],
        std_mask=[0.5],
        resize=None
    ):
        self.mean_img = mean_img
        self.mean_mask = mean_mask
        self.std_img = std_img
        self.std_mask = std_mask
        self.resize = resize

    def __call__(self, comp=None, real=None, mask=None):
        if comp is not None:
            comp = F.to_tensor(comp)
            if self.resize is not None:
                comp = F.resize(comp, self.resize)
            comp = F.normalize(comp, self.mean_img, self.std_img)
        if real is not None:
            real = F.to_tensor(real)
            if self.resize is not None:
                real = F.resize(real, self.resize)
            real = F.normalize(real, self.mean_img, self.std_img)
        if mask is not None:
            mask = F.to_tensor(mask)
            if self.resize is not None:
                mask = F.resize(mask, self.resize)
            mask = F.normalize(mask, self.mean_mask, self.std_mask)

        comp = comp * mask + real * (1 - mask)
        comp = rearrange(comp, 'c h w -> h w c')
        real = rearrange(real, 'c h w -> h w c')
        mask = rearrange(mask, 'c h w -> h w c')
        return comp, real, mask


if __name__ == '__main__':
    dataset = Iharmony4Dataset('dataset/ihm4/Hday2night', True, 256)
    data = next(iter(dataset))
    print(data.keys())
    print(data['comp'].shape)
    print(data['real'].shape)
    print(data['mask'].shape)