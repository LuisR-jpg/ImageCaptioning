from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

import pandas as pd
from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list , read_json

MAX_DIM = 299

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    
    shape = np.array(image.size , dtype = np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim
    
    new_shape = (shape * scale).astype(int)
    image  = image.resize(new_shape)
    
    return image

class RandomRotation:
    def __init__(self , angles=[0,90,180,270]):
        self.angles = angles
    
    def __call__(self,x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle , expand =True)
    
train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['filename']), val['caption'][0])
                      for val in ann]
        self.annot1 = [(self._process(val['filename']), val['caption'][1])
                      for val in ann] ###
        self.annot.extend(self.annot1)###
        self.tokenizer = BertTokenizer.from_pretrained(
            "sagorsarker/bangla-bert-base", do_lower=True)
        if mode == 'validation':
            self.annot = self.annot[-1830:]
        if mode == 'training':
            self.annot = self.annot

        self.max_length = max_length + 1
        
    def _process(self, image_id):
        val = str(image_id)
        return val
    
    def __len__(self):
        return len(self.annot)
    
    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask
    
class CLEFCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()
        self.root = root
        self.transform = transform
        self.annot = [(filename + '.jpg', caption) for filename, caption in zip(ann['ID'].values, ann['caption'])]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower=True)
        self.max_length = max_length + 1
        if mode == 'validation': self.annot = self.annot[: len(self.annot) // 2]
        if mode == 'test': self.annot = self.annot[len(self.annot) // 2:]
        #self.annot = self.annot[: len(self.annot) // 100] #TODO delete this
        
    def _process(self, image_id):
        val = str(image_id)
        return val
    
    def __len__(self):
        return len(self.annot)
    
    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask

def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'ImageCLEFmedical_Caption_2023_train_images', 'train')
        train_file = os.path.join(config.dir, 'train_labels.csv')
        data = CLEFCaption(train_dir, pd.read_csv(train_file, sep = '\t'), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data
    elif mode in ['validation', 'test']:
        val_dir = os.path.join(config.dir, 'ImageCLEFmedical_Caption_2023_valid_images', 'valid')
        val_file = os.path.join(config.dir, 'valid_labels.csv')
        data = CLEFCaption(val_dir, pd.read_csv(val_file, sep = '\t'), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode=mode)
        return data
    else:
        raise NotImplementedError(f"{mode} not supported")