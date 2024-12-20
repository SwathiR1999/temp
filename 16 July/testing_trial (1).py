# -*- coding: utf-8 -*-
"""testing_trial.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vc6DpsonZivrT-RSNMc3z_d7TNbrVgDZ
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/ImageBind finetune/ImageBind-LoRA

!pip install -r requirements.txt --quiet

import os
current_directory = os.getcwd()

batch_size=8

import torch
import logging
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

logging.basicConfig(level=logging.INFO, force=True)

lora = True
linear_probing = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True

assert not (linear_probing and lora), \
    "Linear probing is a subset of LoRA training procedure for ImageBind. " \
    "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    lora_factor = batch_size / 0.07
else:
    lora_factor = 1

image_paths=[]
audio_paths=[]
image_names=[]
audio_names=[]
file_names=[]

test_path = current_directory+"/new_test_data2/"
for file_name in os.listdir(current_directory+"/new_test_data2"):
  file_names.append(file_name[:-4])
for file_name_temp in set(file_names):
  image_paths.append(test_path + file_name_temp + ".jpg")
  image_names.append(file_name_temp + ".jpg")
  audio_paths.append(test_path + file_name_temp + ".wav")
  audio_names.append(file_name_temp + ".wav")

model = imagebind_model.imagebind_huge(pretrained=True)

if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        modality_names=[ModalityType.AUDIO, ModalityType.VISION]))

    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints/lora/", postfix="_last")

    if load_head_post_proc_finetuned:
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints/lora/", postfix="_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints/lora/", postfix="_last")
elif linear_probing:
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints/lora/", postfix="_last")

model.eval()
model.to(device)

inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, to_tensor=True),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

vision_embeddings = embeddings[ModalityType.VISION]
audio_embeddings = embeddings[ModalityType.AUDIO]

gt_audio_to_img =[]
gt_img_to_audio =[]

pred_audio_to_img =[]
pred_img_to_audio =[]


print("Query: Audio")
count_correct=0
count_correct_top5=0
for audio_index, audio_query_embedding in enumerate(audio_embeddings):
  similarity_scores = torch.softmax(vision_embeddings @ audio_query_embedding.T * (lora_factor if lora else 1), dim=-1)
  most_relevant_image_index = torch.argmax(similarity_scores)
  most_relevant_image_path = image_paths[most_relevant_image_index]

  gt_audio_to_img.append(image_names[audio_index])
  pred_audio_to_img.append(image_names[most_relevant_image_index])

  sorted_similarity_scores=similarity_scores.argsort(dim=-1,descending=True)

  if audio_names[audio_index][:-4]==image_names[most_relevant_image_index][:-4]:
    count_correct+=1
  if audio_index in sorted_similarity_scores[:5]:
    count_correct_top5+=1

print("acc(%)_top1:",(count_correct/len(set(file_names)))*100)
print("acc(%)_top5:",(count_correct_top5/len(set(file_names)))*100)



print("Query: Image")
count_correct=0
count_correct_top5=0
for image_index, image_query_embedding in enumerate(vision_embeddings):
  similarity_scores = torch.softmax(audio_embeddings @ image_query_embedding.T * (lora_factor if lora else 1), dim=-1)
  most_relevant_audio_index = torch.argmax(similarity_scores)
  most_relevant_audio_path = audio_paths[most_relevant_audio_index]

  gt_img_to_audio.append(audio_names[image_index])
  pred_img_to_audio.append(audio_names[most_relevant_audio_index])

  sorted_similarity_scores=similarity_scores.argsort(dim=-1,descending=True)
  if image_names[image_index][:-4]==audio_names[most_relevant_audio_index][:-4]:
    count_correct+=1
  if image_index in sorted_similarity_scores[:5]:
    count_correct_top5+=1

print("acc(%)_top1:",(count_correct/len(set(file_names)))*100)
print("acc(%)_top5:",(count_correct_top5/len(set(file_names)))*100)

TP=0
TN=0
FP=0
FN=0
for idx in range(len(gt_audio_to_img)):
  if gt_audio_to_img[idx] == pred_audio_to_img[idx]:
    TP+=1
    TN+=1
  else:
    FP+=1
    FN+=1

precison=  TP/(TP+FP)
recall= TP/(TP+FN)
f1= 2*precison*recall/(precison + recall)

f1

TP=0
TN=0
FP=0
FN=0
for idx in range(len(gt_img_to_audio)):
  if gt_img_to_audio[idx] == pred_img_to_audio[idx]:
    TP+=1
    TN+=1
  else:
    FP+=1
    FN+=1

precison=  TP/(TP+FP)
recall= TP/(TP+FN)
f1= 2*precison*recall/(precison + recall)

f1

# !pip install tensorflow --quiet

# import tensorflow as tf

# for event in tf.compat.v1.train.summary_iterator('lightning_logs/version_0/events.out.tfevents.1720944858.14e37d38128c.1871.0'):
#     for value in event.summary.value:
#         if value.HasField('simple_value'):
#             print(f'{value.tag}: {value.simple_value}')

from sklearn.metrics import precision_score, recall_score, f1_score

# Ground truth and predictions for image-to-audio retrieval
ground_truth = ['Audio A', 'Audio B', 'Audio C', 'Audio D', 'Audio E', 'Audio F', 'Audio G', 'Audio H']
predictions = ['Audio A', 'Audio F', 'Audio C', 'Audio G', 'Audio E', 'Audio B', 'Audio G', 'Audio H']

TP=0
TN=0
FP=0
FN=0
for idx in range(len(ground_truth)):
  if ground_truth[idx] == predictions[idx]:
    TP+=1
    TN+=1
  else:
    FP+=1
    FN+=1

precison=  TP/(TP+FP)
recall= TP/(TP+FN)
f1= 2*precison*recall/(precison + recall)

TP, FP, TN, FN

precison, recall, f1