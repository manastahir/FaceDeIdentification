class Config():
    def __init__(self, base_dir):
        self.image_source_dir=f'{base_dir}/data/Dataset/original'

        self.output_dir=f'{base_dir}/experiments/experiment_1'

        self.classifier_weights=f'{base_dir}/data/weights.pth'

        self.checkpoint_dir='training_checkpoints'

        self.use_trained_model_weights=False

        self.iterations=80000

        self.batch_size=32

        self.steps=None

        self.learning_rate=1e-4

        self.beta1=0.5

        self.beta2=0.99

        self.lamda_step=[5e-7, 1e-6, 2e-6]

        self.lamda_step_iteration=[50000, 100000, 150000]

        self.sample_step=1000

        self.p_loss_layers=[3, 33, 71, 153, 157]

        self.a=[1, 1, 1, 1, 3e-3, 1e-2]

        self.generator_workers=8

        self.source_shape=256

        self.target_shape=224

        self.alpha=5

        self.sigma=2

        self.rotation_range=5

        self.zoom_amount=5

        self.flip=50

        self.random_state=0
        
        self.img_mean = [131.0912, 103.8827, 91.4953]

        self.device='cuda'

        self.show_summary = False
