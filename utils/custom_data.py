import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import os


class FashionMNIST(Dataset):
    def __init__(self, file, dir, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv(os.path.join(dir, file))
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
