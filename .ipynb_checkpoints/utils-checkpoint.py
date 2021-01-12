import torch
import tensorflow as tf
import numpy as np 

def init_weights(layer):
    if(type(layer) == torch.nn.Conv2d):
        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generator(dataset, starting_idx):
    
    for i in range(starting_idx, len(dataset)):
        yield dataset[i]
        
def get_Dataset(train_sequence, starting_idx):
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=(tf.float32, tf.float32, tf.float32),
                                             output_shapes=((None, 256, 256, 3),
                                                            (None, 256, 256, 3),
                                                            (None, 256, 256, 3)),
                                             args=(train_sequence, starting_idx)
                                            )

    return dataset.prefetch(tf.data.AUTOTUNE)

def train_step(model, inputs, loss1, loss2, optim1, optim2):
    s, t, r, rep, lam = inputs
    y = 1-lam[:, :, 0, 0]
    
    optim1.zero_grad()
    outputs = model([s, r, rep, lam])  
    total_loss = loss1(outputs, (r, t, y)).mean()
    total_loss.backward()
    optim1.step()
    
    y = lam[:, :, 0, 0]
    optim2.zero_grad()
    outputs = model([s, r, rep, lam]) 
    l2_loss = loss2(outputs, (r, t, y)).mean()
    l2_loss.backward()
    optim2.step()
    
    return [torch.cat([outputs[0][0], outputs[1][0]], dim=0),
            torch.cat([outputs[0][1], outputs[1][1]], dim=0),
            torch.cat([outputs[0][2], outputs[1][2]], dim=0),
            total_loss.data.item(), l2_loss.data.item()]
    
    
    