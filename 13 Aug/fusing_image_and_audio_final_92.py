import torch
import numpy as np
import random

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import h5py
import numpy as np

with h5py.File('train_val_vision_embeddings.h5', 'r') as f:
    image_embeddings = f['train_val_vision_embeddings'][:]

with h5py.File('train_val_audio_embeddings.h5', 'r') as f:
    audio_embeddings = f['train_val_audio_embeddings'][:]

labels = np.load('train_val_labels_inputs.npy')

import torch
import numpy as np

def l2_normalize(embeddings):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms

    if isinstance(embeddings, torch.Tensor):
        normalized_embeddings = normalized_embeddings.numpy()

    return normalized_embeddings

# image_embeddings = l2_normalize(image_embeddings)
# audio_embeddings = l2_normalize(audio_embeddings)

image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
audio_embeddings = torch.tensor(audio_embeddings, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(image_embeddings, audio_embeddings, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

class DualAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(DualAttention, self).__init__()
        self.image_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.audio_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_embeddings, audio_embeddings):
        img_attn_output, _ = self.image_attn(image_embeddings, image_embeddings, image_embeddings)
        img_attn_output = self.norm(img_attn_output + image_embeddings)

        audio_attn_output, _ = self.audio_attn(audio_embeddings, audio_embeddings, audio_embeddings)
        audio_attn_output = self.norm(audio_attn_output + audio_embeddings)

        combined_attn_output, _ = self.cross_modal_attn(img_attn_output, audio_attn_output, audio_attn_output)
        combined_attn_output = self.norm(combined_attn_output + img_attn_output)

        return combined_attn_output

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.dual_attention = DualAttention(input_dim, num_heads)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, image_embeddings, audio_embeddings):
        combined_embeddings = self.dual_attention(image_embeddings.unsqueeze(1), audio_embeddings.unsqueeze(1))

        combined_embeddings = self.transformer_encoder(combined_embeddings)

        combined_embeddings = self.norm(combined_embeddings.mean(dim=1))

        combined_embeddings = combined_embeddings + image_embeddings

        x = self.fc(combined_embeddings)

        return x


input_dim = image_embeddings.shape[1]  
num_classes = 13 
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)

import sys

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    batch_count = 0
    for image_embeds, audio_embeds, targets in dataloader:
        batch_count += 1
        sys.stdout.write(f"\rBatch {batch_count}/{len(dataloader)}")
        sys.stdout.flush()
        optimizer.zero_grad()
        outputs = model(image_embeds, audio_embeds)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")



with h5py.File('test_vision_embeddings.h5', 'r') as f:
    test_image_embeddings = f['test_vision_embeddings'][:]

with h5py.File('test_audio_embeddings.h5', 'r') as f:
    test_audio_embeddings = f['test_audio_embeddings'][:]

test_labels = np.load('test_labels_inputs.npy')

# test_image_embeddings = l2_normalize(test_image_embeddings)
# test_audio_embeddings = l2_normalize(test_audio_embeddings)

import torch

test_image_embeddings = torch.tensor(test_image_embeddings, dtype=torch.float32)
test_audio_embeddings = torch.tensor(test_audio_embeddings, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(test_image_embeddings, test_audio_embeddings, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

from sklearn.metrics import precision_score, recall_score, f1_score

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for image_embeds, audio_embeds, labels in test_dataloader:
        outputs = model(image_embeds, audio_embeds)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')