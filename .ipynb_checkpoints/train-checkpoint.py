import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from comet_ml import Experiment,ExistingExperiment

from config import Config
from datagenerator import AugmentedImageSequence
from loss import LossFunction, MSEloss
from model import Generator, Discriminator, AdvserialAutoEncoder
from classifier import ft_model
import utils

from sklearn.metrics import accuracy_score

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from parallel import DataParallelModel, DataParallelCriterion

def main():
    cp = Config('..')

    ts_file = os.path.join(cp.output_dir, "training_stats.json")
    checkpoint_dir = f'{cp.output_dir}/{cp.checkpoint_dir}'
    lamda = 2e-6
    
    if not os.path.isdir(cp.output_dir):
        os.makedirs(cp.output_dir)
        os.makedirs(f'{checkpoint_dir}')

    if cp.use_trained_model_weights:
        print("** use trained model weights **")

        training_stats = json.load(open(ts_file))
        
        iteration = training_stats["iteration"]
        starting_idx = training_stats["idx"]
        experiment_key = training_stats["experiment_key"]
      
    else:
        iteration=0
        starting_idx=0
        training_stats = {"lamda" : lamda, "iteration": iteration}
        experiment_key = None

    print(f"backup files to {cp.output_dir}")
    os.system('cp -r . ' + cp.output_dir)

    print("** Summary Writer **")
    logdir = f'{cp.output_dir}/logs'
    writer = SummaryWriter(logdir)

    if(experiment_key is None):
        experiment = Experiment(api_key='MMehTpWSVV3FsYXMlNUCADXTQ',
                            project_name='FaceDeIdentification pytorch-256', 
                            workspace='anastahir', 
                            log_code=True,
                            auto_histogram_tensorboard_logging=True,
                            auto_histogram_gradient_logging=True,
                            auto_histogram_activation_logging=True,
                            auto_output_logging=False,
                            log_graph=True,
                        )
        experiment.add_tag('experiment_1')
        training_stats['experiment_key'] = experiment.get_key()
        print(f'Stated new experiment with key {experiment.get_key()}')
    else:
        experiment = ExistingExperiment(api_key='MMehTpWSVV3FsYXMlNUCADXTQ', 
                                    previous_experiment=f'{experiment_key}')
        print(f'Resumed experiment with key {experiment.get_key()}')
            

    print("** create image generators **")

    params = {
            "batch_size": cp.batch_size,
            "source_size":cp.source_shape,
            "mean": cp.img_mean,
            "source_image_dir":cp.image_source_dir,
            "steps":cp.steps,
            "alpha":cp.alpha,
            "sigma":cp.sigma,
            "rotation_range":cp.rotation_range,
            "zoom_amount":cp.zoom_amount,
            "shuffle":False,
            "random_state": cp.random_state,
            "flip": cp.flip,
            "idx": starting_idx*cp.batch_size,
        }

    train_sequence = AugmentedImageSequence(
        params
    )
    
    Dataset = torch.utils.data.DataLoader(train_sequence, cp.batch_size, shuffle=False, num_workers=18, pin_memory=True)

    print("** generate face classifier **")
    feature_extractor = ft_model(
        weights_path=cp.classifier_weights
        )

    for name, param in feature_extractor.named_parameters():
        try:
            param.requires_grad = False
        except Exception as e: 
            print(f"Could not freeze weights {name}: {e}")

    feature_extractor.to(cp.device)

    print("** generate loss function **")
    loss_function = LossFunction(
        classifier=feature_extractor,
        alpha=cp.a,
        lamda=lamda
        )
    
    generator_loss = DataParallelCriterion(loss_function, device_ids=[0, 1])
    discriminator_loss = DataParallelCriterion(MSEloss(), device_ids=[0, 1])
    
    print("** initialize model **")
    generator = Generator()
    discriminator = Discriminator()

    print(f"** set Checkpoint dir to: {checkpoint_dir} **")

    if cp.use_trained_model_weights: 
        print("** load model **")
        generator.load_state_dict(torch.load(f'{checkpoint_dir}/generator.pth'))
        discriminator.load_state_dict(torch.load(f'{checkpoint_dir}/discriminator.pth'))
        

    else:
        generator.apply(utils.init_weights)
        discriminator.apply(utils.init_weights)
    
    model = AdvserialAutoEncoder(generator=generator,
                                discriminator=discriminator,
                                feature_extractor=feature_extractor
                                )
    model.to(cp.device)
    model = DataParallelModel(model, device_ids=[0,1])
    
    generator_optim = torch.optim.Adam(generator.parameters(), lr=cp.learning_rate, betas=(cp.beta1, cp.beta2))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=cp.learning_rate, betas=(cp.beta1, cp.beta2)) 

    if(cp.show_summary):
        summary(generator, [(3, cp.source_shape, cp.source_shape), (2048,)], device=cp.device)
        summary(discriminator, (3, cp.source_shape, cp.source_shape), device=cp.device)
    
    gen_param = utils.count_parameters(generator)
    dis_param = utils.count_parameters(discriminator)

    print(f'Number of parameters in Generator {gen_param}')
    print(f'Number of parameters in Discriminator {dis_param}')
    
    print(f"** start training **")
    print(f"** Sample step {cp.sample_step} lamda {lamda} starting iteration {iteration} index {starting_idx}\
    generator length {len(train_sequence)} total iterations {cp.iterations}**")

    img_mean = np.array(cp.img_mean) / 255.0
    pbar = tqdm(total=cp.iterations, initial=iteration)
    
    with experiment.train():
        while(iteration < cp.iterations):
            for idx, inputs in enumerate(Dataset):
                source, target, real, lam = inputs
                
                source = source.to(cp.device)
                target = target.to(cp.device)
                real = real.to(cp.device)
                lam = lam.to(cp.device)
                inputs = [source, target, real, lam]
                raw, mask, masked, t_loss, l2_loss = utils.train_step(model, inputs, generator_loss, discriminator_loss, generator_optim, discriminator_optim)

                iteration+=1
                metrics = {'total_loss':t_loss, 'L2_loss': l2_loss}
                writer.add_scalars('Metrics',metrics,global_step=iteration)
                experiment.log_metrics(metrics, step=iteration)
                pbar.update(1)

                if(iteration % 100 ==0):
                    i = np.random.randint(0, int(cp.batch_size/2-1))
                    
                    diff = torch.abs(masked[i]-target[i])
                    experiment.log_histogram_3d(generator.decoder_dense.weight.cpu().detach().numpy())

                    experiment.log_image(target[i].cpu().detach(), 'target', image_channels="first", step=iteration)
                    experiment.log_image(raw[i].cpu().detach()   , 'raw',    image_channels="first", step=iteration)
                    experiment.log_image(masked[i].cpu().detach(), 'masked', image_channels="first", step=iteration)
                    experiment.log_image(diff.cpu().detach(),      'diff',   image_channels="first", step=iteration)
                
                if(iteration % cp.sample_step == 0):
                    writer.flush()
                    
                    training_stats["iteration"] = iteration
                    training_stats["idx"] = idx

                    with open(ts_file, "w") as logger:
                        json.dump(training_stats, logger)

                    torch.save(generator.state_dict(), f'{checkpoint_dir}/generator.pth')
                    torch.save(discriminator.state_dict(), f'{checkpoint_dir}/discriminator.pth')
            
                if(iteration % cp.iterations == 0):
                    break

            starting_idx = 0
            train_sequence.idx = starting_idx
            train_sequence.on_epoch_end()
            
        print("** done! **")

if __name__ == "__main__":
    main()