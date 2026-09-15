### Simple dataset that loads HDR images and normalizes them to [0, 1] range

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from glob import glob

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset

IMG_DIRECTORIES = {
    "RAISE": "RAISE/raw/*.exr",
    "fivek": "fivek_dataset/raw_photos/*/photos/*.exr",
    "LSMI-galaxy": "LSMI/galaxy/*/*.exr",
    "LSMI-nikon": "LSMI/nikon/*/*.exr",
    "LSMI-sony": "LSMI/sony/*/*.exr",
    "SID": "SID/Sony/long/*.exr",
    "multiRAW-huawei_p30pro": "multiRAW/huawei_p30pro/raw/*.exr",
    "multiRAW-iphone_xsmax": "multiRAW/iphone_xsmax/raw/*.exr",
    "multiRAW-oneplus_5t": "multiRAW/oneplus_5t/raw/*.exr",
    "zoom_raw": "zoom_raw/*/*/*.exr",
    "nikon_raw": "nikon_raw/long/*.exr",
    "canon_raw": "canon_raw/Canon/long/*.exr",
    "hdrplusdata": "hdrplusdata/20171106/bursts/*/*.exr",
    "ppr": "ppr/raw/*.exr",
}


class HDRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the HDR images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(
            os.path.join(root_dir, "*.hdr")
        )  # Assuming HDR files have .hdr extension

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        # Load HDR image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")

        # Convert BGR to RGB
        image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = image * 0.18 / np.median(np.unique(image))

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        # Ensure the image has a reasonable range
        if image.max() < 1 or image.min() > 0.1:
            return self.__getitem__(np.random.randint(0, len(self)))

        return image, img_path


class RAWDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            root_dir (str): Directory with all the HDR images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.image_paths = []
        for dir_pattern in IMG_DIRECTORIES.values():
            self.image_paths.extend(glob(dir_pattern))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        # Load HDR image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")

        # Convert BGR to RGB
        image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = image * 0.18 / np.median(np.unique(image))

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        return image, img_path


class CropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the HDR images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(
            os.path.join(root_dir, "*.exr")
        )  # Assuming HDR files have .hdr extension

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        # Load HDR image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")

        # Convert BGR to RGB
        image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = image * 0.18 / np.median(np.unique(image))

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        # if image.max() < 32:
        #    image = image * 32 / image.max()

        return image, img_path
