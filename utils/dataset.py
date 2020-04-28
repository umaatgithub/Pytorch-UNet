from os.path import splitext
from os import listdir
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import logging

class BasicDataset(Dataset):
    img_w = 1024
    img_h = 360
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

#        self.labels = {
#            'road'  :  [128, 64,128],
#            'sidewalk' :  [244, 35,232],
#            'parking' :  [250,170,160],
#            'rail track' : [230,150,140],
#            'building' : [ 70, 70, 70],
#            'wall' : [102,102,156],
#            'fence' : [190,153,153],
#            'guard rail' : [180,165,180],
#            'bridge' : [150,100,100],
#            'tunnel' : [150,120, 90],
#            'pole' : [153,153,153],
#            'polegroup' : [153,153,153],
#            'traffic light' : [250,170, 30],
#            'traffic sign' : [220,220,  0],
#            'vegetation' : [107,142, 35],
#            'terrain' : [152,251,152],
#            'sky' : [ 70,130,180],
#            'person': (220, 20, 60),
#            'rider' : [255,  0,  0],
#            'car' : [0,  0,142],
#            'truck' : [ 0,  0, 70],
#            'bus' : [ 0, 60,100],
#            'caravan' : [  0,  0, 90],
#            'trailer' : [  0,  0,110],
#            'train' : [  0, 80,100],
#            'motorcycle' : [ 0,  0,230],
#            'bicycle' : [119, 11, 32],
#            'license plate' : [  0,  0,142] 
#        }        

       
        self.file_names = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.file_names)} examples')

    def __len__(self):
        return len(self.file_names)

    @classmethod
    def preprocess_image(self, img_nd, scale):
#        w, h = pil_img.size
#        newW, newH = int(scale * w), int(scale * h)
#        assert newW > 0 and newH > 0, 'Scale is too small'
#        pil_img = pil_img.resize((newW, newH))
#
        #img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
#        if img_trans.max() > 1:
#            img_trans = img_trans / 255

    	#KITTI dataset
        img_crop_dim = tuple(map(lambda i, j: i - j, img_trans.shape[1:3], (BasicDataset.img_h, BasicDataset.img_w)))
        img_crop = img_trans[:, img_crop_dim[0]//2 : -(img_crop_dim[0]//2+img_crop_dim[0]%2), img_crop_dim[1]//2 : -(img_crop_dim[1]//2+img_crop_dim[1]%2)]
        #img_tran = img_trans[:,7:-8,109:-109]
#        im = Image.fromarray(img_tran.transpose((2, 1, 0))
#        im.show()

        return img_crop

    @classmethod
    def preprocess_mask(self, img_nd, scale):

        #img_nd = np.array(pil_img)

        labels = {
            0  : [  0,  0,  0],
            1  : [128, 64,128],
            2  : [244, 35,232],
            3  : [250,170,160],
            4  : [230,150,140],
            5  : [ 70, 70, 70],
            6  : [102,102,156],
            7  : [190,153,153],
            8  : [180,165,180],
            9  : [150,100,100],
            10 : [150,120, 90],
            11 : [153,153,153],
            12 : [153,153,153],
            13 : [250,170, 30],
            14 : [220,220,  0],
            15 : [107,142, 35],
            16 : [152,251,152],
            17 : [ 70,130,180],
            18 : [255,  0,  0],
            19 : [  0,  0,142],
            20 : [  0,  0, 70],
            21 : [  0, 60,100],
            22 : [  0,  0, 90],
            23 : [  0,  0,110],
            24 : [  0, 80,100],
            25 : [  0,  0,230],
            26 : [119, 11, 32],
            27 : [  0,  0,142]
        }

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
        #if img_mask.max() > 1:
        #    img_mask = img_mask / 255
        img_crop_dim = tuple(map(lambda i, j: i - j, img_mask.shape[1:3], (BasicDataset.img_h, BasicDataset.img_w)))
        img_mask = img_mask[:, img_crop_dim[0] // 2: -(img_crop_dim[0] // 2 + img_crop_dim[0] % 2),
                   img_crop_dim[1] // 2: -(img_crop_dim[1] // 2 + img_crop_dim[1] % 2)]
            
        return img_mask
    
    def __getitem__(self, i):
        file_name = self.file_names[i]
        #print(file_name)
        img_file = glob(self.imgs_dir + file_name + '*')
        mask_file = glob(self.masks_dir + file_name + '*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        img = np.array(Image.open(img_file[0]))
        mask = np.array(Image.open(mask_file[0]))

#        assert img.size == mask.size, \
#            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess_image(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

def main():
    print("hello world!")
    dataset = BasicDataset('../../data/training/image/', '../../data/training/semantic_rgb/')
    dataloader = DataLoader(dataset, batch_size=1)
    data = next(iter(dataset))
    print('Image size : ', data['image'].shape)
    print('Mask size : ', data['mask'].shape)
#    print(data['image'].numpy())
    
if __name__ == "__main__":
    main()
