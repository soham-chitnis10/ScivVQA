from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import os
from PIL import Image
from typing import NamedTuple, Dict
from torchvision import transforms
from torchvision.transforms import ToTensor

class ImageDict(NamedTuple):
    image: Image.Image
    data: Dict


class SciVQADataset(Dataset):
    def __init__(self, json_file, image_root):
        self.df = pd.read_json(json_file)
        self.image_root = image_root

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = item['image_file']
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert("RGB")
        return ImageDict(image=image, data=item.to_dict())


