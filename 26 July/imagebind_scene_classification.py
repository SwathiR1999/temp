# -*- coding: utf-8 -*-
"""imagebind_scene_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Brw2yMsxM_Ae5869yEm0Y9SHKCLU2yjI
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/ImageBindFinetuning_new

!pip install -r requirements.txt --quiet

import os
current_directory = os.getcwd()

import torch
import logging
import data
from models import imagebind_model
from models.imagebind_model import ModalityType

logging.basicConfig(level=logging.INFO, force=True)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

model = imagebind_model.imagebind_huge(pretrained=True)

model.eval()
model.to(device)

train_path = current_directory+"/new_data/"
train_path_img = train_path + "images/"
train_path_audio = train_path + "audio/"

labels = {}
for lb_idx,label in enumerate(os.listdir(train_path_img)):
  labels[label] = lb_idx

image_paths = []
audio_paths = []
label_list = []
for label in os.listdir(train_path_img):
  train_path_img_cat = train_path_img + label
  train_path_audio_cat = train_path_audio + label
  for img_file_name in os.listdir(train_path_img_cat):
    img_file_path = train_path_img_cat + "/" + img_file_name
    audio_file_name = img_file_name.split(".")[0] + ".wav"
    audio_file_path = train_path_audio_cat + "/" + audio_file_name
    image_paths.append(img_file_path)
    audio_paths.append(audio_file_path)
    label_list.append(labels[label])

image_paths.sort()
audio_paths.sort()
label_list.sort()

# image_paths = image_paths[:10]
# audio_paths = audio_paths[:10]
# label_list = label_list[:10]

# inputs = {
#     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, to_tensor=True),
#     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
# }

# torch.save(inputs, 'inputs.pth')

loaded_inputs = torch.load('inputs.pth', map_location=torch.device('cpu'))

# import h5py

# with h5py.File('vision_embeddings.h5', 'w') as h5f:
#   num_embeddings = loaded_inputs['vision'].shape[0]
#   embedding_size = 1024
#   dataset = h5f.create_dataset('vision_embeddings', (num_embeddings, embedding_size), dtype='f')
#   for i in range(num_embeddings):
#         # Calculate the embedding for the current input
#         embd = model({'vision': torch.unsqueeze(loaded_inputs['vision'][i], dim=0)})

#         # Write the embedding to the dataset
#         dataset[i] = embd['vision'].detach().cpu().numpy()

#         # Free the memory used by the current embedding
#         del embd
#         # torch.cuda.empty_cache()  # Clear cached memory if using GPU

import numpy as np
file_path = 'vision_embeddings.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as h5f:
    # Access the dataset
    dataset = h5f['vision_embeddings']

    # Load the data into a NumPy array
    vision_embeddings = np.array(dataset)

print("Embeddings loaded from 'embeddings.h5'")
print(vision_embeddings.shape)  # Print the shape of the loaded embeddings

import h5py
with h5py.File('audio_embeddings.h5', 'w') as h5f:
  num_embeddings = loaded_inputs['audio'].shape[0]
  embedding_size = 1024
  dataset = h5f.create_dataset('audio_embeddings', (num_embeddings, embedding_size), dtype='f')
  for i in range(num_embeddings):
        # Calculate the embedding for the current input
        embd = model({'audio': torch.unsqueeze(loaded_inputs['audio'][i], dim=0)})

        # Write the embedding to the dataset
        dataset[i] = embd['audio'].detach().cpu().numpy()

        # Free the memory used by the current embedding
        del embd
        # torch.cuda.empty_cache()  # Clear cached memory if using GPU

import numpy as np
file_path = 'audio_embeddings.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as h5f:
    # Access the dataset
    dataset = h5f['audio_embeddings']

    # Load the data into a NumPy array
    audio_embeddings = np.array(dataset)

print("Embeddings loaded from 'embeddings.h5'")
print(audio_embeddings.shape)  # Print the shape of the loaded embeddings

import numpy as np

vision_embeddings_np = embeddings[ModalityType.VISION].cpu().numpy()
audio_embeddings_np = embeddings[ModalityType.AUDIO].cpu().numpy()

np.save('vision_embeddings.npy', vision_embeddings_np)
np.save('audio_embeddings.npy', audio_embeddings_np)

vision_embeddings_np_loaded = np.load('vision_embeddings.npy')
audio_embeddings_np_loaded = np.load('audio_embeddings.npy')

vision_embeddings = torch.tensor(vision_embeddings_np_loaded)
audio_embeddings = torch.tensor(audio_embeddings_np_loaded)

vision_embeddings.shape

audio_embeddings.shape

X = []
y = []
for i in range(vision_embeddings.shape[0]):
  #concatenated_embedding = np.concatenate((vision_embeddings[i], audio_embeddings[i]))
  concatenated_embedding = vision_embeddings[i] + audio_embeddings[i]
  # np.save(f'concatenated_embedding_{i}.npy', concatenated_embedding)
  X.append(concatenated_embedding)
  y.append(label_list[i])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)

from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear')  #'rbf
svm_classifier.fit(X_train, y_train)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Assuming y_pred and y_test are already defined
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")