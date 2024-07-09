# -*- coding: utf-8 -*-
"""testing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SIC40sxlRdBnqMfh8N3a-1aQebD_ThKF
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/ImageBind finetune/ImageBind-LoRA

!pip install -r requirements.txt --quiet

import os
current_directory = os.getcwd()

import torch
import logging
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

logging.basicConfig(level=logging.INFO, force=True)

lora = False
linear_probing = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True

assert not (linear_probing and lora), \
    "Linear probing is a subset of LoRA training procedure for ImageBind. " \
    "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    lora_factor = 12 / 0.07
else:
    lora_factor = 1

image_paths = ["test_data/00088_2.jpg", "test_data/00095.jpg", "test_data/00158_4.jpg",
               "test_data/00169_3.jpg", "test_data/00220.jpg", "test_data/00222_3.jpg"]
audio_paths = ["test_data/00088_2.wav", "test_data/00095.wav", "test_data/00158_4.wav",
               "test_data/00169_3.wav", "test_data/00220.wav", "test_data/00222_3.wav"]

for i in range(len(image_paths)):
    image_paths[i] = current_directory + "/" + image_paths[i]
for i in range(len(audio_paths)):
    audio_paths[i] = current_directory + "/" + audio_paths[i]

model = imagebind_model.imagebind_huge(pretrained=True)

# if lora:
#     model.modality_trunks.update(
#         LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
#                                         modality_names=[ModalityType.TEXT, ModalityType.VISION]))

#     LoRA.load_lora_modality_trunks(model.modality_trunks,
#                                    checkpoint_dir=".checkpoints/lora/", postfix="_last")

#     if load_head_post_proc_finetuned:
#         load_module(model.modality_postprocessors, module_name="postprocessors",
#                     checkpoint_dir=".checkpoints/lora/", postfix="_last")
#         load_module(model.modality_heads, module_name="heads",
#                     checkpoint_dir=".checkpoints/lora/", postfix="_last")
# elif linear_probing:
#     load_module(model.modality_heads, module_name="heads",
#                 checkpoint_dir="./.checkpoints/lora/", postfix="_last")

model.eval()
model.to(device)

# Load data
#audio_query_paths = [current_directory + "/test_data/00222_3.wav"]  # Example: Choose one audio query

inputs_audio = {
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

inputs_image = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

# Generate embeddings for audio query
with torch.no_grad():
    embeddings_audio = model(inputs_audio)
with torch.no_grad():
    embeddings_image = model(inputs_image)

vision_embeddings = embeddings_image[ModalityType.VISION]
audio_query_embedding = embeddings_audio[ModalityType.AUDIO]

count_crt=0
for audio_index, audio_query_embedding in enumerate(embeddings_audio['audio']):

  similarity_scores = torch.softmax(vision_embeddings @ audio_query_embedding.T * (lora_factor if lora else 1), dim=-1)

  most_relevant_image_index = torch.argmax(similarity_scores)

  most_relevant_image_path = image_paths[most_relevant_image_index]

  if most_relevant_image_index==audio_index:
    count_crt+=1

  print("Audio query:",audio_paths[audio_index].split('/')[-1] )
  print("Most relevant image for audio query:", most_relevant_image_path.split('/')[-1])

count_crt

for index, audio_query_emb in enumerate(embeddings_audio['audio']):
  print(index, audio_query_emb)

vision_embeddings.shape

vision_embeddings

audio_query_embedding.T.shape

torch.softmax(vision_embeddings @ audio_query_embedding.T,dim=-1)

torch.softmax(vision_embeddings @ audio_query_embedding.T,dim=0)

torch.softmax(vision_embeddings @ audio_query_embedding.T,dim=1)