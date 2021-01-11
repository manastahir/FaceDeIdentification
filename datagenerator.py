import torch
from torch.utils.data import Dataset

import numpy as np
import os
import cv2
import tqdm

from augmentation import ImageAugmentation

class AugmentedImageSequence(Dataset):
    def __init__(self, params):

        self.source_image_dir = params['source_image_dir']
        self.batch_size = params['batch_size']

        self.source_size = params['source_size']
        self.target_size = params['target_size']
        self.mean = params['mean'] 

        self.shuffle = params['shuffle']
        self.random_state = params['random_state']
        steps = params['steps']

        self.augmenter = ImageAugmentation(params)
        self.prepare_dataset()

        if steps is None:
            self.steps = int(len(self.img_paths) / float(self.batch_size))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        
        img_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            
        images = [cv2.resize(cv2.imread(os.path.join(self.source_image_dir, path)), (self.source_size, self.source_size)) for path in img_paths]
            
        real = np.array([(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)-self.mean)/255 for img in images])

        real = self.augmenter.random_flip(real)

        source = self.augmenter.trasnform(real.copy())

        target = source.copy()

        return (torch.Tensor(np.transpose(source, (0,3,1,2)).astype('float32')), 
                torch.Tensor(np.transpose(target, (0,3,1,2)).astype('float32')), 
                torch.Tensor(np.transpose(real,   (0,3,1,2)).astype('float32')),
                )

    def prepare_dataset(self):
        self.img_paths = os.listdir(self.source_image_dir)
        np.random.seed(self.random_state)
        np.random.shuffle(self.img_paths)
    
    def on_epoch_end(self):
        self.prepare_dataset()
        if(self.shufle):
            self.random_state+=1
            np.random.seed(self.random_state)
            np.random.shuffle(self.img_paths)