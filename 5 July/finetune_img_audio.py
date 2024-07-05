# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/ImageBind finetune/
# !git clone --recurse-submodules -j8 https://github.com/fabawi/ImageBind-LoRA.git --quiet

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/ImageBind finetune/ImageBind-LoRA/

!pip install -r requirements.txt

import os
from PIL import Image
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

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

class ImageAudioDataset(Dataset):
    def __init__(self, image_root_dir, audio_root_dir, transform=None):
        self.image_root_dir = image_root_dir
        self.audio_root_dir = audio_root_dir
        self.transform = transform

        # List of all image files and their corresponding audio files
        self.image_files = []
        self.audio_files = []
        self.labels = []

        for class_name in os.listdir(image_root_dir):
            class_image_dir = os.path.join(image_root_dir, class_name)
            class_audio_dir = os.path.join(audio_root_dir, class_name)
            if os.path.isdir(class_image_dir):
                for image_name in os.listdir(class_image_dir):
                    if image_name.endswith('.png') or image_name.endswith('.jpg'):
                        image_path = os.path.join(class_image_dir, image_name)
                        audio_path = os.path.join(class_audio_dir, image_name.rsplit('.', 1)[0] + '.wav')
                        if os.path.exists(audio_path):
                            self.image_files.append(image_path)
                            self.audio_files.append(audio_path)
                            self.labels.append(class_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        audio, sample_rate = torchaudio.load(audio_path)
        return image, audio, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Initialize the dataset
image_dir = 'data/images'
audio_dir = 'data/audio'
# Create the dataset
dataset = ImageAudioDataset(image_dir, audio_dir, transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Function to display an image and its corresponding audio waveform
def display_sample(image, audio, label, idx):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Display image
    ax[0].imshow(image.permute(1, 2, 0).numpy())
    ax[0].set_title(f"Image {idx} - Class: {label}")
    ax[0].axis('off')

    # Display audio waveform
    ax[1].plot(audio.squeeze().numpy())
    ax[1].set_title(f"Audio {idx} - Class: {label}")
    ax[1].set_xlabel("Sample")
    ax[1].set_ylabel("Amplitude")

    plt.show()

# Visualize a few samples
num_samples_to_view = 5
for i, (image, audio, label) in enumerate(dataloader):
    if i >= num_samples_to_view:
        break
    display_sample(image[0], audio[0], label, i)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

class ImageBindTrain(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = imagebind_model.imagebind_huge(pretrained=True)

        if self.hparams.lora:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)

            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=self.hparams.lora_rank))
                                                                              #layer_idxs=self.hparams.lora_layer_idxs,
                                                                              #modality_names=self.hparams.lora_modality_names))
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)

            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        images, audios, labels = batch

        feats_image = self.model({"vision": images})
        feats_audio = self.model({"audio": audios})

        feats_image_tensor = list(feats_image.values())[0]
        feats_audio_tensor = list(feats_audio.values())[0]

        cos_sim = F.cosine_similarity(feats_image_tensor[:, None, :], feats_audio_tensor[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        self.log(f"{mode}_loss", nll, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_postprocessors, module_name="postprocessors", checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads", checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            save_module(self.model.modality_heads, module_name="heads", checkpoint_dir=self.hparams.lora_checkpoint_dir)

from lightning.pytorch import Trainer

# Initialize the model
model = ImageBindTrain(max_epochs=2, batch_size=32, lr=5e-4, weight_decay=1e-4, lora=True, lora_rank=4)

# Initialize the trainer
trainer = Trainer(max_epochs=500, gpus=1)

# Train the model
trainer.fit(model, train_loader, val_loader)
