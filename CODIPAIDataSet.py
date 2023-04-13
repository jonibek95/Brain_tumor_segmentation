import os
from typing import Dict, List
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import cv2
from PIL import Image

# FOR .npy IMAGES

# class CODIPAIDataSet(Dataset):
#     def __init__(self, imgs_dir: str, masks_dir: str):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir

#         self.ids = [
#             os.path.splitext(fi)[0]
#             for fi in os.listdir(self.masks_dir)
#             if not fi.startswith(".")
#         ]

#     def __len__(self) -> int:
#         return len(self.ids)

#     def __getitem__(self, i) -> Dict[List[torch.FloatTensor], List[torch.FloatTensor]]:
#         idx = self.ids[i]
#         img_idx = idx.split("_")[-1]
#         img_name = idx.split("_")[0]

#         img_path = os.path.join(self.imgs_dir, f"{img_name}_{img_idx}.npy")
#         mask_path = os.path.join(self.masks_dir, f"{img_name}_{img_idx}.npy")

#         img_file = glob(img_path)
#         mask_file = glob(mask_path)

#         assert (
#             len(mask_file) == 1
#         ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
#         assert (
#             len(img_file) == 1
#         ), f"Either no image or multiple images found for the ID {idx}: {img_file}"

#         image = np.load(img_file[0], allow_pickle=True)
#         masks = np.load(mask_file[0], allow_pickle=True)

#         masks = np.array([mask for mask in masks])
#         image = np.transpose(image, (2, 0, 1))
#         masks = np.transpose(masks, (2, 0, 1))

#         return {
#             "image": torch.tensor(image, dtype=torch.float32),
#             "mask": torch.tensor(masks, dtype=torch.float32),
#         }


# FOR .png IMAGES

class CODIPAIDataSet(Dataset):
    def __init__(self, imgs_dir: str, masks_dir: str):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.ids = [
            os.path.splitext(fi)[0]
            for fi in os.listdir(self.masks_dir)
            if not fi.startswith(".")
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        idx = self.ids[i]

        img_path = os.path.join(self.imgs_dir, f"{idx}.png")
        mask_path = os.path.join(self.masks_dir, f"{idx}.png")

        img_file = glob(img_path)
        mask_file = glob(mask_path)

        assert (
                len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert (
                len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"
        
        image = Image.open(img_file[0]).convert('RGB')
        mask = Image.open(mask_file[0])
        
        image = np.array(image)
        mask = np.array(mask)

#         image = np.load(img_file[0], allow_pickle=True)
# #         print('1: ', image.shape)
#         mask = np.load(mask_file[0], allow_pickle=True)
#         mask = mask.squeeze()

        image = np.transpose(image, (2, 0, 1))
        mask = mask / 255.
#         mask = np.transpose(mask, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }
if __name__ == "__main__":
    import pandas as pd

    images = pd.read_pickle("images.pickle")
    mapper = pd.read_pickle("mapper.pickle")

    rgb2gray = eval(mapper.loc[0, "rgb2gray"])
    class2rgb = eval(mapper.loc[0, "class2rgb"])
    cls_id = eval(mapper.loc[0, "cls_id"])
    rgb2cls = eval(mapper.loc[0, "rgb2cls"])
    gray2class = eval(mapper.loc[0, "gray2class"])

    ddd = CODIPAIDataSet("./data/data/val_test/train/image", "./data/data/val_test/train/mask")
    train_loader = DataLoader(
        ddd, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    d = next(iter(train_loader))
    print(d["image"][0].shape, d["mask"][0].shape)
    print(np.unique(d["image"]))
    print(d["mask"].shape)
