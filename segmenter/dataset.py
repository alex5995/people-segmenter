import sys
sys.path.insert(0, 'segmenter')

import os
import numpy as np
import torch
from PIL import Image
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PeopleDataset(object):

    def __init__(self, root, transforms):

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Training_Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Ground_Truth"))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "Training_Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.root, "Ground_Truth", self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    os.system('git clone https://github.com/VikramShenoy97/Human-Segmentation-Dataset.git')

    os.remove('Human-Segmentation-Dataset/Ground_Truth/.DS_Store')
    os.remove('Human-Segmentation-Dataset/Training_Images/.DS_Store')

    masks = os.listdir('Human-Segmentation-Dataset/Ground_Truth')
    print('Masks:', len(masks))

    imgs = os.listdir('Human-Segmentation-Dataset/Training_Images')
    print('Images:', len(imgs))