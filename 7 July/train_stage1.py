# -*- coding: utf-8 -*-
"""train_stage1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CyTrWzLC78pbayPSUJeUIbK_6N-d353V
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/ImageBind finetune/ImageBind-LoRA

!pip install -r requirements.txt

#!git clone https://github.com/enthought/mayavi.git

import numpy as np
print(np.__version__)

import os

num_workers = os.cpu_count()
print(f"Number of CPU cores: {num_workers}")

import torch

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    print(f"CUDA Device ID: {device_id}")
    print(f"CUDA Device Name: {device_name}")
else:
    print("CUDA is not available")

import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import data

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

self_contrast=True
batch_size=16
num_workers= os.cpu_count()
lora_modality_names_123 = ["vision", "audio"]
LOG_ON_STEP = True
LOG_ON_EPOCH = True
lora=False
full_model_checkpointing=True
full_model_checkpoint_dir="./.checkpoints/full"
lora_checkpoint_dir="./.checkpoints/lora"
device_name="cuda:0" if torch.cuda.is_available() else "cpu"
max_epochs=2
gradient_clip_val=1.0
loggers=None
linear_probing=True

class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42,
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95),
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False
                 ):
        super().__init__()
        assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. " \
            "Linear probing stores params in lora_checkpoint_dir"
        self.save_hyperparameters()

        # Load full pretrained ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        if lora:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)

            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                              layer_idxs=lora_layer_idxs,
                                                                              modality_names=lora_modality_names))
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir)

            # Load postprocessors & heads
            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
        elif linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        data_a, class_a, data_b, class_b = batch

        # class_a is always "vision" according to ImageBind
        feats_a = [self.model({class_a[0]: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        if self.hparams.self_contrast:
            feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
            feats_tensors = [feats_a_tensor, feats_a_b_tensor]
            temperatures = [1, self.hparams.temperature]
            contrast = ["self", "cross"]
        else:
            feats_a_b_tensor = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
            feats_tensors = [feats_a_b_tensor]
            temperatures = [self.hparams.temperature]
            contrast = ["cross"]

        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2
            # Logging loss
            self.log(mode + "_loss_" + contrast[feats_idx], nll, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

        self.log(mode + "_loss", dual_nll, prog_bar=True,
                 on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        return dual_nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            # Save LoRA checkpoint
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            # Save postprocessors & heads
            save_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            # Save postprocessors & heads
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)

class ImageAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_size=0.8, random_seed=42, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(os.path.join(root_dir, 'images')) if os.path.isdir(os.path.join(root_dir, 'images', d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.audio_paths = []
        for cls in self.classes:
            cls_image_dir = os.path.join(root_dir, 'images', cls)
            cls_audio_dir = os.path.join(root_dir, 'audio', cls)
            for filename in os.listdir(cls_image_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths.append((os.path.join(cls_image_dir, filename), cls))
            for filename in os.listdir(cls_audio_dir):
                if filename.endswith('.wav'):
                    self.audio_paths.append((os.path.join(cls_audio_dir, filename), cls))

        # Split dataset
        self.train_image_paths, self.test_image_paths = train_test_split(self.image_paths, train_size=train_size, random_state=random_seed)
        self.train_audio_paths, self.test_audio_paths = train_test_split(self.audio_paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.image_paths = self.train_image_paths
            self.audio_paths = self.train_audio_paths
        elif split == 'test':
            self.image_paths = self.test_image_paths
            self.audio_paths = self.test_audio_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return min(len(self.image_paths), len(self.audio_paths))

    def __getitem__(self, index):
        img_path, class_text = self.image_paths[index]
        audio_path, _ = self.audio_paths[index]

        # Load and transform image
        images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=False)
        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        # Load and transform audio
        audios = data.load_and_transform_audio_data([audio_path], self.device)

        # Load and transform text
        texts = data.load_and_transform_text([class_text], self.device)

        return images, ModalityType.VISION, audios, ModalityType.AUDIO#, texts, ModalityType.TEXT

contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

train_datasets = []
test_datasets = []

train_datasets.append(ImageAudioDataset(
            root_dir=os.getcwd()+"/data/", split="train",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if self_contrast else 1)))
test_datasets.append(ImageAudioDataset(
            root_dir=os.getcwd()+"/data/", split="test",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if self_contrast else 1)))

train_dataset = train_datasets[0]
test_dataset = test_datasets[0]

train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=num_workers,
    )
val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=num_workers,
    )

lora_layer_idxs = {}
lora_modality_names = []
modalities = ["vision", "text", "audio", "thermal", "depth", "imu"]
for modality_name in lora_modality_names_123:
    if modality_name in modalities:
        modality_type = getattr(ModalityType, modality_name.upper())
        #lora_layer_idxs[modality_type] = getattr(args, f'lora_layer_idxs_{modality_name}', None)
        # if not lora_layer_idxs[modality_type]:
        #     lora_layer_idxs[modality_type] = None
        lora_layer_idxs[modality_type] = None
        lora_modality_names.append(modality_type)
    else:
        raise ValueError(f"Unknown modality name: {modality_name}")

model = ImageBindTrain(max_epochs=max_epochs, batch_size=batch_size,
                           num_workers=num_workers, self_contrast=self_contrast,
                           lora=lora, lora_checkpoint_dir=lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=linear_probing)

if full_model_checkpointing:
        checkpointing = {"enable_checkpointing": full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_loss", dirpath=full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="min")]}
else:
        checkpointing = {"enable_checkpointing": full_model_checkpointing,}

trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=1 if ":" not in device_name else [int(device_name.split(":")[1])], deterministic=True,
                      max_epochs=max_epochs, gradient_clip_val=gradient_clip_val,
                      logger=loggers if loggers else None, **checkpointing)

trainer.fit(model, train_loader, val_loader)