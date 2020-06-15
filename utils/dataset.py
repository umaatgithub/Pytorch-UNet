from os.path import splitext
from os import listdir
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import logging

from utils.load_config import data

class BasicDataset(Dataset):
    img_h = data['training']['image']['height']
    img_w = data['training']['image']['width']

    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.file_names = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.file_names)} examples')

    def __len__(self):
        return len(self.file_names)

    @classmethod
    def preprocess_image(self, img_nd, scale):
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        # Normalize
        if img_trans.max() > 1:
            img_trans = img_trans / 255

    	#Crop image height and width to fit network input size
        img_crop_dim = tuple(map(lambda i, j: i - j, img_trans.shape[1:3], (BasicDataset.img_h, BasicDataset.img_w)))
        if img_crop_dim[0] > 0:
            start_h = img_crop_dim[0] // 2
            end_h = -(img_crop_dim[0] // 2 + img_crop_dim[0] % 2)
        else:
            start_h = 0
            end_h = -1
        if img_crop_dim[0] > 0:
            start_w = img_crop_dim[1] // 2
            end_w = -(img_crop_dim[1] // 2 + img_crop_dim[1] % 2)
        else:
            start_w = 0
            end_w = -1
        img_crop = img_trans[:, start_h:end_h, start_w:end_w]

        return img_crop

    @classmethod
    def preprocess_mask(self, img_nd, scale):
        labels = data['class']['colors']
        #print(labels)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_mask = np.zeros((img_nd.shape[0], img_nd.shape[1], 1))
        for x in range(img_nd.shape[0]):
            for y in range(img_nd.shape[1]):
                for key, value in labels.items():
                    if (img_nd[x,y,:] == np.array(value)).all():
                        img_mask[x,y,0] = key
                        break
                        
        # HWC to CHW
        img_mask = img_mask.transpose((2, 0, 1))

        # Crop mask to input image size
        img_crop_dim = tuple(map(lambda i, j: i - j, img_mask.shape[1:3], (BasicDataset.img_h, BasicDataset.img_w)))
        if img_crop_dim[0] > 0:
            start_h = img_crop_dim[0] // 2
            end_h = -(img_crop_dim[0] // 2 + img_crop_dim[0] % 2)
        else:
            start_h = 0
            end_h = -1
        if img_crop_dim[0] > 0:
            start_w = img_crop_dim[1] // 2
            end_w = -(img_crop_dim[1] // 2 + img_crop_dim[1] % 2)
        else:
            start_w = 0
            end_w = -1
        img_mask = img_mask[:, start_h:end_h, start_w:end_w]
            
        return img_mask
    
    def __getitem__(self, i):
        file_name = self.file_names[i]

        img_file = glob(self.imgs_dir + file_name + '*')
        mask_file = glob(self.masks_dir + file_name + '*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {file_name}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {file_name}: {mask_file}'

        if data['training']['preprocess']['flag'] :
            img = np.array(Image.open(img_file[0]))
            mask = np.array(Image.open(mask_file[0]))
            #assert img.size == mask.size, \
            #f'Image and mask {file_name} should be the same size, but are {img.size} and {mask.size}'
            img = self.preprocess_image(img, self.scale)
            mask = self.preprocess_mask(mask, self.scale)
        else :
            img = np.load(img_file[0])
            mask = np.load(mask_file[0])

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

def main():
    print("hello world!")
    dataset = BasicDataset('../../data/training/image/', '../../data/training/semantic_rgb/')
    dataloader = DataLoader(dataset, batch_size=1)
    data = next(iter(dataloader))
    print('Image size : ', data['image'].shape)
    print('Mask size : ', data['mask'].shape)
    
if __name__ == "__main__":
    main()
