import json
import os

import albumentations as A
import cv2 as cv
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CarDatasets(Dataset):
    path_images = './data/vkcv2022-contest-02-carplates/data'

    def __init__(self, name_json: str = 'train.json', train_size: float = 0.9,
                 type_dataset: str = 'train', width: int = 640, height: int = 640, transform: A = None):

        with open(os.path.join(self.path_images, name_json)) as js_file:
            data = json.load(js_file)

        if type_dataset == 'train':
            self.all_data, _ = train_test_split(data, train_size=train_size, random_state=17)
        elif type_dataset == 'val':
            _, self.all_data = train_test_split(data, train_size=train_size, random_state=17)
        else:
            print("Неверный тип датасета! М.б. 'train' / 'val'")

        self.width = width
        self.height = height
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        annot_data = self.all_data[idx]

        try:
            image = cv.cvtColor(
                cv.imread(os.path.join(self.path_images, annot_data['file'])),
                cv.COLOR_BGR2RGB
            ).astype(np.float32)
        except Exception:
            idx = 0
            annot_data = self.all_data[idx]
            image = cv.cvtColor(
                cv.imread(os.path.join(self.path_images, annot_data['file'])),
                cv.COLOR_BGR2RGB
            ).astype(np.float32)
            print("Can't open image!")

        image_resized = cv.resize(image, (self.width, self.height))
        image_resized /= 255.0

        image_width = image.shape[1]
        image_height = image.shape[0]

        annotation = annot_data['nums']

        boxes = []

        for box in annotation:
            boxes.append(
                [
                    (min([box[0] for box in box['box']])/image_width)*self.width,
                    (min([box[1] for box in box['box']])/image_height)*self.height,
                    (max([box[0] for box in box['box']])/image_width)*self.width,
                    (max([box[1] for box in box['box']])/image_height)*self.height
                ]
            )

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(np.ones(shape=(boxes.shape[0]), dtype=np.int64), dtype=torch.int64)

        if self.transform is not None:
            try:
                sample = self.transform(image=image_resized, bboxes=boxes, labels=labels)
                image_resized = sample['image']
                boxes = torch.Tensor(sample['bboxes'])
            except Exception:
                print("Can't transform!")

        image_resized = transforms.ToTensor()(image_resized)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image_resized, target


def collate_fn(batch):
    return tuple(zip(*batch))









