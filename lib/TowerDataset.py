import os
import pandas as pd
import numpy as np
import torch
import random
from PIL import Image

from torch.utils.data import Dataset

# Pytorch Dataset Class, inherited from torch.utils.data.Dataset
class TowerDataset(Dataset):
       
    def __init__(self, root, transforms=None):
        self.root = root
        self.img_labels = pd.read_csv(self.root + 'finaltotal.csv')
        self.img_labels = self.img_labels.groupby('image').agg(list).reset_index()
        
        self.img_list   = pd.read_csv(self.root + 'images.csv')
        self.transforms  = transforms

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        
        img_path =  self.img_list.loc[self.img_list['Filename'] == self.img_labels.iloc[idx, 0], 'Path'].iloc[0]
        img = Image.open(img_path)
        
        num_objs = len(self.img_labels.iloc[idx, 19])

        
        labels = [] 
        for item in self.img_labels.iloc[idx, 19]:
            labels.append(item)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        boxes = [] 
        for i in range(num_objs):
            xmax = self.img_labels.iloc[idx, 15][i]
            ymax = self.img_labels.iloc[idx, 16][i]
            xmin = self.img_labels.iloc[idx, 17][i]
            ymin = self.img_labels.iloc[idx, 18][i]
        
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        
            boxes.append([xmin,ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  
        
        image_id = torch.tensor([idx])
        
        iscrowd = torch.zeros(num_objs, dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 
             
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
