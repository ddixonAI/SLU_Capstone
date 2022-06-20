
"""
Code mostly adopted from here: https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch/tree/main/UNET
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import torch
import time
from torch.utils.data import Dataset

""" Seeding the randomness. """
def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_dir(path):
    """Creates new directories if they don't already exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):

    train_x = sorted(glob(os.path.join(path, "datasets", "training", "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "datasets", "training", "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "datasets", "test", "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "datasets", "test", "test", "mask", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_image(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extract the name """
        name = x.split('/')[-1].split('\\')[-1].split('.')[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        """ Defining augmentation parameters and resizing """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        index = 0
        for image, mask in zip(X, Y):
            i = cv2.resize(image, size)
            m = cv2.resize(mask, size)

            tmp_image_name = f'{name}_{index}.png'
            tmp_mask_name = f'{name}_{index}.png'

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

def prepare_data(data_path):
    """ Seeding """
    np.random.seed(31)

    """ Load the data """
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f'Train Image Count: X: {len(train_x)}, Y: {len(train_y)}')
    print(f'Test Image Count: X: {len(test_x)}, Y: {len(test_y)}')

    """ Create new directories to store images """
    create_dir('new_data/train/image/')
    create_dir('new_data/train/mask/')
    create_dir('new_data/test/image/')
    create_dir('new_data/test/mask/')

    """ Perform augmentation """
    augment_image(train_x, train_y, "new_data/train/", augment=True)
    augment_image(test_x, test_y, "new_data/test/", augment=False)