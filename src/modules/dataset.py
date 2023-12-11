import os
import torch
from torch.utils.data import Dataset
import cv2
import json
from PIL import Image
from torchvision import transforms


class GetDataSet(Dataset):
    """
    class for preparing annotations
    """
    def __init__(self, path_to_imgs, path_to_json):
        self.path_to_imgs = path_to_imgs
        self.path_to_json = path_to_json
        self.img_ids = os.listdir(path_to_imgs)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        img_id = os.path.splitext(image_name)[0]
        img_format = os.path.splitext(self.img_ids[idx])[1]
        img = cv2.imread(os.path.join(self.path_to_imgs, img_id + img_format))
        load_json = json.load(open(os.path.join(self.path_to_json, img_id + ".json")))
        left_lung = [[point['x'], point['y']] for point in load_json['annotations'][0]['polygon']['path']]
        right_lung = [[point['x'], point['y']] for point in load_json['annotations'][1]['polygon']['path']]
        lungs = {'left_lung': left_lung,
                 'right_lung': right_lung}
        return img, lungs, image_name

    def __len__(self):
        return len(self.img_ids)


class CustomDatasetWithContours(Dataset):
    """
    class for processing and providing image-mask-contour pairs.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.Resize((576, 576)), transforms.ToTensor()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        mask_path = self.dataframe.iloc[idx, 1]
        contours_path = self.dataframe.iloc[idx, 2]

        img = Image.open(str(img_path)).convert('RGB')
        mask = Image.open(str(mask_path)).convert('L')
        contours = Image.open(str(contours_path)).convert('L')
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            contours = self.transform(contours)
        target = torch.cat([mask, contours], dim=0)
        return img, target

