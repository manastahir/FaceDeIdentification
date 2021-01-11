import torch
from torch.utils.data import Dataset

import numpy as np
import os
import cv2
import tqdm

from imgaug import augmenters as iaa
import imgaug as ia


class AugmentedImageSequence(Dataset):
    def __init__(self, params):

        self.source_image_dir = params['source_image_dir']

        self.mean = params['mean'] 

        self.shuffle = params['shuffle']
        self.random_state = params['random_state']
        steps = params['steps']
         
        self.augmenter1 = iaa.Sequential([
        iaa.Resize(params['source_size']),
        iaa.Fliplr(0.5),
        ])
            
        self.augmenter2 = iaa.Sequential([
        iaa.Resize(params['source_size']),
        iaa.Affine(scale=(1-params["zoom_amount"] /100, 1+params["zoom_amount"] /100),
                  rotate=(-params["rotation_range"], params["rotation_range"])),
        iaa.ElasticTransformation(params["alpha"], params["sigma"])
        ])
        
        self.prepare_dataset(params['idx'])

        if steps is None:
            self.steps = len(self.img_paths)
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        
        path = self.img_paths[idx]
        image = (cv2.cvtColor(cv2.imread(f'{self.source_image_dir}/{path}'),cv2.COLOR_BGR2RGB)-self.mean)/255
        
        real = self.augmenter1.augment_image(image)
        source = self.augmenter2.augment_image(real.copy())
        target = source.copy()
        
        lam = np.random.beta(0.2, 0.2, size=(1, 1, 1))
        
        return (torch.Tensor(np.transpose(source, (2,0,1)).astype('float32')), 
                torch.Tensor(np.transpose(target, (2,0,1)).astype('float32')), 
                torch.Tensor(np.transpose(real,   (2,0,1)).astype('float32')),
                torch.Tensor(lam.astype('float32')),
                )

    def prepare_dataset(self, idx):
        img_paths = os.listdir(self.source_image_dir)
        self.img_paths = img_paths[idx:]
        [self.img_paths.append(path) for path in img_paths]
        
        np.random.seed(self.random_state)
        np.random.shuffle(self.img_paths)
    
    def on_epoch_end(self):
        self.prepare_dataset()
        if(self.shufle):
            self.random_state+=1
            np.random.seed(self.random_state)
            np.random.shuffle(self.img_paths)