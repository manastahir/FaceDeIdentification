import os
import pickle
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from sklearn.metrics import accuracy_score

from config import Config
from data_generator import AugmentedImageSequence
from loss import LossFunction
from classifier import Vggface2_ResNet50
from model import Generator, Discriminator, AdvserialAutoEncoder

from utils import write_images, write_model_layers, write_log, wma, resize, get_dist_data_gen

def main():
    cp = Config('/content')

    checkpoint_prefix = f'{cp.output_dir}/{cp.checkpoint_dir}'
    strategy = tf.distribute.MirroredStrategy()

    ts_file = os.path.join(cp.output_dir, "training_stats.json")
    print(f'*****Using {strategy}*****')
    print(f'*****Setting checkpoint_prefix to {checkpoint_prefix}*****')


    if not os.path.isdir(cp.output_dir):
        os.makedirs(cp.output_dir)
        os.makedirs(f'{cp.output_dir}/{cp.checkpoint_dir}')

    if cp.use_trained_model_weights:
        print("** use trained model weights **")

        training_stats = json.load(open(ts_file))

        lamda = training_stats["lamda"]
        iteration = training_stats["iteration"]
        starting_idx = training_stats["idx"]


    else:
        iteration=0
        starting_idx=0
        lamda = 2e-6
        training_stats = {"iteration": iteration, "idx":starting_idx}

    os.system('cp -r . ' + cp.output_dir)

    logdir = f'{cp.output_dir}/logs'
    summary_writer = tf.summary.create_file_writer(logdir)
    
    print('*****Making Data Genertor*****')
    face_descriptor = Vggface2_ResNet50(weights=cp.classifier_weights,
                                            trainable=False,
                                            output_layers=[-3])
    params = {
            "batch_size":cp.batch_size,
            "source_size":cp.source_shape,
            "target_size":cp.target_shape,
            "mean": cp.img_mean,
            "source_image_dir":cp.image_source_dir,
            "steps":cp.steps,
            "alpha":cp.alpha,
            "sigma":cp.sigma,
            "rotation_range":cp.rotation_range,
            "zoom_amount":cp.zoom_amount,
            "shift_range":cp.shift_range,
            "shuffle":False,
            "random_state": cp.random_state,
            "flip": cp.flip,
            "face_descriptor":face_descriptor,
            "idx":1
        }

    train_sequence = AugmentedImageSequence(
        params
    )
    
    print('*****Making Loss Function*****')
    loss_classifier = Vggface2_ResNet50(weights=cp.classifier_weights,
                                        trainable=False,
                                        output_layers=cp.p_loss_layers)
    
    print('*****Making Maps Model*****')
    maps_model = Vggface2_ResNet50(weights=cp.classifier_weights,
                                   trainable=False,
                                   output_layers=cp.p_loss_layers+[-3,-1])

    print('*****Making Main Model*****')
    with strategy.scope():
        generator = Generator()    
        discriminator = Discriminator()

        generator_optim = tf.keras.optimizers.Adam(learning_rate=cp.learning_rate, beta_1=cp.beta1, beta_2=cp.beta2) 
        discriminator_optim = tf.keras.optimizers.Adam(learning_rate=cp.learning_rate, beta_1=cp.beta1, beta_2=cp.beta2) 


        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optim,
                                      discriminator_optimizer=discriminator_optim,
                                      generator=generator,
                                      discriminator=discriminator)

        checkpoint_manager = tf.train.CheckpointManager(
                            checkpoint, checkpoint_prefix, max_to_keep=1)

        if cp.use_trained_model_weights:
            print("** load model **")
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

    loss_function = LossFunction(classifier = loss_classifier)
    model = AdvserialAutoEncoder(generator=generator,
                                discriminator=discriminator,
                                loss=loss_function,
                                gen_optim=generator_optim,
                                disc_optim=discriminator_optim,
                                alpha=cp.a,
                                lamda=lamda)



    if(cp.show_summary):
        _, _ = generator([np.random.rand(1,256,256,3),np.random.rand(1,2048)])
        _ = discriminator(np.random.rand(1,256,256,3))

        generator.summary()
        discriminator.summary()

    print(f'*****Iteration: {iteration}, Starting-idx: {starting_idx}, Total-Iterations:{cp.iterations}, Lambda:{lamda}, Num-replicas:{strategy.num_replicas_in_sync}*****')
    print('starting training')
    
    pbar = tqdm(total=cp.iterations, initial=iteration)

    with strategy.scope():
        while(iteration < cp.iterations):
            dist_data_gen = get_dist_data_gen(train_sequence, strategy)
            for source, target, real, lam, description in dist_data_gen:

                raw, mask, masked, losses = model.distributed_train_step(strategy, (source, target, real, description, lam))
                Rdisc, Fdisc, LrR, LrM, LpR, LpM, gR, gM, m = losses
                Rdisc, Fdisc, LrR, LrM, LpR, LpM, gR, gM, m = Rdisc.numpy(), Fdisc.numpy(), LrR.numpy(), LrM.numpy(), LpR.numpy(), LpM.numpy(), gR.numpy(), gM.numpy(), m.numpy()

                iteration+=1
                raw    = raw.values[0].numpy()
                mask   = mask.values[0].numpy()
                masked = masked.values[0].numpy()
                target = target.values[0].numpy()

                m_activations = maps_model.predict(masked)
                t_activations = maps_model.predict(target)

                prep1 = np.abs(t_activations[0]-m_activations[0]).mean()
                prep2 = np.abs(t_activations[1]-m_activations[1]).mean()
                prep3 = np.abs(t_activations[2]-m_activations[2]).mean()
                prep4 = np.abs(t_activations[3]-m_activations[3]).mean()
                prep5 = np.abs(t_activations[5]-m_activations[5]).mean()

                real_identity = np.argmax(t_activations[6], axis=1)
                changed_identity = np.argmax(m_activations[6], axis=1)

                accuracy = accuracy_score(real_identity, changed_identity)

                pbar.update(1)
                if(iteration % 100 ==0):
                    i = np.random.randint(0, cp.batch_size/(strategy.num_replicas_in_sync)-1)
                    diff =  target[i:i+1]-masked[i:i+1]
                    data = {
                        "target" : source[i:i+1]+img_mean,
                        "raw" : raw[i:i+1]+img_mean,
                        "mask" : mask[i:i+1]+img_mean,
                        "masked" : masked[i:i+1]+img_mean,
                    }
                    write_images(data, (256, 256, 3), iteration, summary_writer)

                write_log({'disc_loss': Rdisc, 'prep': LpR+LpM, 'rec': LrR+LrM, 'mask':m, 'grad':gM+gR, 'FakeDisc': Fdisc,
                            'Accuracy': accuracy, '112x112_maps': prep1, '55x55_maps': prep2, '28x28_maps': prep3,
                            '7x7_maps': prep4, '1x1_maps': prep5}, iteration, summary_writer)

                if(iteration % cp.sample_step == 0):
                    summary_writer.flush()

                    training_stats["iteration"] = iteration
                    training_stats["idx"] = idx

                    with open(ts_file, "w") as logger:
                        json.dump(training_stats, logger)

                    checkpoint_manager.save(checkpoint_number=0)
            train_sequence.on_epoch_end()

if __name__ == "__main__":
    main()