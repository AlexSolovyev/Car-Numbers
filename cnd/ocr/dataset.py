import imageio
import numpy as np

from torch.utils.data import Dataset

class OcrDataset(Dataset):
    def __init__(self, data_path, transforms=None):

        self.data = []
        self.target = []
        for img_file in data_path:
            self.target.append(
                img_file.split('/')[-1].split('_')[0],
            )

            self.data.append(img_file)

        self.transforms = transforms


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = imageio.imread(self.data[idx])
        text = self.target[idx]
        if self.transforms is not None:
            img = self.transforms(img)

        return {
            "image": img,
            "text": text,
        }
