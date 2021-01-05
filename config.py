class Config():
    def __init__(self, base_dir):
        self.image_source_dir=f'../data/Dataset/original/'

        self.output_dir=f'../experiments/experiment_1'

        self.classifier_weights=f'../data/weights.h5'

        self.checkpoint_dir='training_checkpoints'

        self.use_trained_model_weights=False

        self.iterations=80000

        self.batch_size=64

        self.steps=None

        self.learning_rate=1e-4

        self.beta1=0.5

        self.beta2=0.99

        self.sample_step=10000

        self.k=10000

        self.window=21

        self.p_loss_layers=[3, 36, 78, 172, 174]

        self.a=[0.5, 0.5, 0.5, 0.5, 3e-3, 1e-2]

        self.generator_workers=8

        self.source_shape=256

        self.target_shape=224

        self.alpha=3

        self.sigma=2

        self.rotation_range=5

        self.zoom_amount=5

        self.shift_range=0

        self.flip=50

        self.random_state=0
        
        self.img_mean = [91.4953, 103.8827, 131.0912]

        self.show_summary = False