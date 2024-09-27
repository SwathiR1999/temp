# -*- coding: utf-8 -*-
"""test_dct_classifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R5sf9jzv2Czy1F5iq-tSLCYHgFZhA6fN
"""

import os
import torch
import torch_dct as dct
import argparse
import random
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda:0"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def array_from_imgdir(imgdir, grayscale=True):
    paths = []
    imgnames = os.listdir(imgdir)
    for imgname in imgnames:
        if imgname == ".DS_Store":
          continue
        paths.append(os.path.join(imgdir, imgname))

    if grayscale:
        def loader(path):
            return transforms.ToTensor()(Image.open(path).convert("L"))

    array = torch.stack(
                Parallel(n_jobs=8)(delayed(loader)(path) for path in paths)
            )

    print('final array shape', array.shape)
    array = (array*2.0) - 1  # scale to [-1, 1]
    return array
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.linear1(x))
        out2 = self.relu(self.linear2(out1))
        out3 = self.linear3(out2)
        return out3

class MyDataset(Dataset):
    def __init__(self, x_data, labels):
        self.x_data = x_data
        self.labels = labels

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = torch.tensor(self.labels[idx])
        return x, y

def load_and_preprocess_images(imgdir):
    # Load and preprocess the images as done during training
    dataset = MyDataset(imgdir)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Placeholder for processed images
    images = []
    for inputs in data_loader:
        # Perform DCT and log transformation
        x_tf = dct.dct_2d(inputs, norm='ortho')
        x_tf = torch.log(torch.abs(x_tf) + 1e-12)
        x_tf = x_tf.squeeze(1).reshape(x_tf.shape[0], -1)
        images.append(x_tf)

    return torch.cat(images)

def test_model(model, valid_loader):
    model.eval()
    valid_acc = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            labels = labels.long()
            preds = torch.argmax(outputs, dim=1)
            valid_acc += (preds == labels).float().mean().item()

    return valid_acc / len(valid_loader)

def main(args):
    # Load the model
    model = LogisticRegression(args.input_size * args.input_size, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))

    # Load and preprocess test dataset
    real_val = array_from_imgdir(
        args.image_root / "test" / "real"
    )
    fake_val = array_from_imgdir(
        args.image_root / "test" / "fake"
    )
    x_val = torch.cat([real_val, fake_val], dim=0)
    y_val = torch.tensor([0.0] * len(real_val) + [1.0] * len(fake_val))
    del real_val, fake_val

    print('feature calculation...')

    x_val_tf = dct.dct_2d(x_val, norm = 'ortho')
    x_val_tf = torch.log(torch.abs(x_val_tf) + 1e-12)

    x_val_tf = x_val_tf.squeeze(1)
    x_val_tf = x_val_tf.reshape(x_val_tf.shape[0], -1)
    print('reshaped...')
    # Normalize using means and stds saved during training
    means = torch.load(args.means_path)
    stds = torch.load(args.stds_path)

    x_test = (x_val_tf - means) / stds

    test_dataset = MyDataset(x_test, y_val)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    test_accuracy = test_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=Path,
        help="Directory containing test images."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='model.pth',
        help="Path to the saved model."
    )
    parser.add_argument(
        "--means_path",
        type=str,
        default='means.pt',
        help="Path to the saved means."
    )
    parser.add_argument(
        "--stds_path",
        type=str,
        default='stds.pt',
        help="Path to the saved standard deviations."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Size of input image"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())