import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
import tqdm
from augmentation import ImageAugmentation

class AugmentedImageSequence(Sequence):
    def __init__(self, params):

        self.source_image_dir = params['source_image_dir']
        self.batch_size = params['batch_size']

        self.source_size = params['source_size']
        self.target_size = params['target_size']
        self.mean = params['mean'] 

        self.shuffle = params['shuffle']
        self.random_state = params['random_state']
        steps = params['steps']
        
        self.face_descriptor = params['face_descriptor']

        self.augmenter = ImageAugmentation(params)
        self.prepare_dataset(params['idx'])

        if steps is None:
            self.steps = int(len(self.img_paths) / float(self.batch_size))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        for path in paths:
            img = cv2.imread(os.path.join(self.source_image_dir, path))
            img = cv2.resize(img, (self.source_size, self.source_size))
            images.append(img)
        
        real = np.array([(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)-self.mean)/255 for img in images])
       
        real = self.augmenter.random_flip(real)

        source = self.augmenter.trasnform(real.copy())

        target = source.copy()
        
        face_description = self.face_descriptor.predict(target)
        
        lam = np.random.beta(0.2, 0.2, size=(self.batch_size, 1, 1, 1))
        
        return (tf.convert_to_tensor(source, dtype=tf.float32), 
                tf.convert_to_tensor(target, dtype=tf.float32), 
                tf.convert_to_tensor(real, dtype=tf.float32),
                tf.convert_to_tensor(lam, dtype=tf.float32),
                tf.convert_to_tensor(face_description, dtype=tf.float32),
               )

    def prepare_dataset(self, idx):
        img_paths = os.listdir(self.source_image_dir)
        self.img_paths = img_paths[self.batch_size*idx:]
        [self.img_paths.append(path) for path in img_paths[:self.batch_size*idx]]
    
    def on_epoch_end(self):
        self.img_paths = os.listdir(self.source_image_dir)
