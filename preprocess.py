from os.path import splitext, exists
from os import listdir,mkdir
from glob import glob
import numpy as np
from PIL import Image

from utils.dataset import BasicDataset
from utils.load_config import data

def main():
    print("Hello")
    assert exists(data['training']['image']['path']), \
        f"Image path {data['training']['image']['path']} does not exist"
    assert exists(data['training']['mask']['path']), \
        f"Mask path {data['training']['mask']['path']} does not exist"

    file_names = [splitext(file)[0] for file in listdir(data['training']['image']['path'])
                  if not file.startswith('.')]

    if not exists(data['training']['image']['pre_path']):
        mkdir(data['training']['image']['pre_path'])
    if not exists(data['training']['mask']['pre_path']):
        mkdir(data['training']['mask']['pre_path'])

    for file_name in file_names:
        img_file = glob(data['training']['image']['path'] + file_name + '*')
        mask_file = glob(data['training']['mask']['path'] + file_name + '*')

        print("Processing file : " + file_name)
        img = np.array(Image.open(img_file[0]))
        mask = np.array(Image.open(mask_file[0]))

        img = BasicDataset.preprocess_image(img, 1.0)
        mask = BasicDataset.preprocess_mask(mask, 1.0)

        np.save(data['training']['image']['pre_path'] + file_name + ".npy", img)
        np.save(data['training']['mask']['pre_path'] + file_name + ".npy", mask)

if __name__ == "__main__" :
    print("Main function")
    main()

