import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def transform_image(image):
    return torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)

class HybridDataset(Dataset):
    def __init__(self):
        vi_dir = r"D:\paper_implimentation\project_root\Hybrid\data\data\vi"
        ir_dir = r"D:\paper_implimentation\project_root\Hybrid\data\data\ir"
        hr_dir = r"D:\paper_implimentation\project_root\Hybrid\data\train_data\HR"
        self.vi_paths = [os.path.join(vi_dir, f) for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg'))]
        self.ir_paths = [os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg'))]
        self.hr_paths = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]
        self.vi_paths.sort()
        self.ir_paths.sort()
        self.hr_paths.sort()
        assert len(self.vi_paths) == len(self.ir_paths) == len(self.hr_paths), "Mismatch in number of images"
        self.transform = transform_image

    def __len__(self):
        return len(self.vi_paths)

    def __getitem__(self, idx):
        vi_img = Image.open(self.vi_paths[idx]).convert('RGB')
        ir_img = Image.open(self.ir_paths[idx]).convert('L')  # Grayscale for IR
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')
        return self.transform(vi_img), self.transform(ir_img), self.transform(hr_img)