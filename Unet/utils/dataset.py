from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import numpy as np
import cv2
import random
import math

CONFIG = {
    'flip':0.5,                 # flip config Probability of flip augmentation
    'hsv': [0.6,(0.5,0.5,0.5)], # hsv config [Probability of HSV augmentation, (hue_gain, saturation_gain, value_gain)]
    'shape': [0.2,(5, 0.1, 5, 0.1)], # Affine config [Probability of affine augmentation, (degree, scale, shear, translate)]
}

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  
    return im

def mapping_func(x, r, l, h):
    x = round(x * r)
    x = min(h,x)
    x = max(l,x)
    return x

class CustomDataset(Dataset):
    def __init__(self, p, train = True, img_size = 512, augment = True, conf = CONFIG):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        if train == True:
            label_root = Path(str(p) + '/train/label')
            img_root = Path(str(p) + '/train/image')
            self.labeldir = glob.glob(str(label_root / '**'), recursive=True)[1:]
            self.imagedir = glob.glob(str(img_root / '**'), recursive=True)[1:]
        else:
            label_root = None
            img_root = Path(str(p) + '/test/image')
            self.labeldir = []
            self.imagedir = glob.glob(str(img_root / '*.*'), recursive=True)[1:]
        self.train = train
        self.img_sz = img_size
        self.augment = augment
        self.conf = conf
        print(len(self.imagedir))
            
    def __getitem__(self, i):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        (img, (h0,w0), (h,w)), label = self.load_image(i), self.load_label(i)

        if self.augment and self.train:
            for k in self.conf:
                img, label = self.imgaug(k, self.conf[k],img, label)
        
        # Transform to single channal img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, img.shape[0], img.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # Formatting labels
        if label.max() > 1:
            label = label / 255

        return img, label

    def __len__(self):
        return len(self.imagedir)

    def imgaug(self, aug_type, aug_conf, im, lb):
        at = aug_type
        ac = aug_conf
        if at == 'flip':
            if ac > random.random():
                r = random.choice([-1,0,1])
                im = cv2.flip(im,r)
                lb = cv2.flip(lb,r)
            return im, lb
        elif at == 'hsv':
            if ac[0] > random.random():
                hgain, sgain, vgain = ac[1]
                im = augment_hsv(im, hgain, sgain, vgain)
            return im, lb
        elif at == 'shape':
            if ac[0] > random.random():
                degree, scale, shear, translate = ac[1]
                height = im.shape[0]
                width = im.shape[1]

                # Center
                C = np.eye(3)
                C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
                C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

                # Rotation and Scale
                R = np.eye(3)
                a = random.uniform(-degree, degree)
                s = random.uniform(1 - scale, 1 + scale)
                R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

                # Shear
                S = np.eye(3)
                S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
                S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

                # Translation
                T = np.eye(3)
                T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
                T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

                # Combined rotation matrix
                M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
                # affine img
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
                # affine label
                lb = cv2.warpAffine(lb, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
            return im, lb

    
    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.imagedir[i]
        im = cv2.imread(f)  # BGR
        assert im is not None, f'Image Not Found {f}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_sz / max(h0, w0)  # ratio
        interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def load_label(self, i):
        # Loads 1 label from dataset index 'i', returns a label
        if len(self.labeldir) < 1 or not self.train:
            return None
        f = self.labeldir[i]
        lb = cv2.imread(f)  # BGR
        assert lb is not None, f'Label Not Found {f}'
        h0, w0 = lb.shape[:2]  # orig hw
        r = self.img_sz / max(h0, w0)  # ratio
        interp = cv2.INTER_NEAREST
        lb = cv2.resize(lb, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return lb

if __name__ == "__main__":
    import os
    dir = r'..\data'
    dir = os.path.abspath(dir)
    print(dir)
    # You can then use the prebuilt data loader. 
    custom_dataset = CustomDataset(dir,conf = CONFIG)
    train_loader = DataLoader(dataset=custom_dataset,
                                            batch_size=1, 
                                            shuffle=True)

    s, (im,lb) = next(enumerate(train_loader))
    im = im[0]

    print(lb[0])
    print(im.numpy().shape)
    cv2.imshow('',im.numpy()[0])
    cv2.waitKey(0)

    print(lb[0].numpy()[0])
    cv2.imshow('',lb[0].numpy()[0])
    cv2.waitKey(0)




