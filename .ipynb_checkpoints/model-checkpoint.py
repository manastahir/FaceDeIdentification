from tensorflow.keras.layers import Layer, SeparableConv2D, Conv2D, LeakyReLU, Reshape, Dense, Flatten, Add, Concatenate, Activation, UpSampling2D
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

class PixelShuffler(Layer):
    def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = data_format
        self.size = size

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
            return out

        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
            return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class Sequential(Model):
    def __init__(self):
        super(Sequential, self).__init__()

        self.k_init = initializers.RandomNormal(mean=0.0, stddev=0.02, seed=0)
    
    def call(self, x):
        msg = f'call has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

class SeperableConvBlock(Sequential):
    def __init__(self, filters, block_id, base_name):
        super(SeperableConvBlock, self).__init__()
        
        self.conv = SeparableConv2D(filters=filters, kernel_size=3, strides=2, use_bias=False,padding='same',
                         depthwise_initializer=self.k_init, pointwise_initializer=self.k_init, 
                         name=f'{base_name}_conv_{block_id}') 
        self.norm = InstanceNormalization(name=f'{base_name}_normalize_{block_id}')
    
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)

        return x

class ConvBlock(Sequential):
    def __init__(self, filters, block_id, base_name, norm=True):
        super(ConvBlock, self).__init__()
        
        self.do_norm = norm

        self.conv = Conv2D(filters=filters, kernel_size=3, strides=2, use_bias=False,padding='same',
                         kernel_initializer=self.k_init,name=f'{base_name}_conv_{block_id}') 
        self.norm = InstanceNormalization(name=f'{base_name}_normalize_{block_id}')
        self.activ = LeakyReLU(name=f'{base_name}_normalize_{block_id}')
    
    def call(self, x):
        x = self.conv(x)
        if (self.do_norm):
            x = self.norm(x)
        x = self.activ(x)

        return x

class UpScale(Sequential):
    def __init__(self, filters, block_id, base_name):
        super(UpScale, self).__init__()

        self.conv = Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer=self.k_init, 
                           use_bias=False, padding='same', name=f'{base_name}_upscale_conv_{block_id}')
        
        self.norm = InstanceNormalization(name=f'{base_name}_upscale_activ_{block_id}')

        self.activ = LeakyReLU(alpha=0.1, name=f'{base_name}_upscale_activ_{block_id}')

        self.reshape = PixelShuffler(name=f'{base_name}_upscale_reshape_{block_id}')
    
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        x = self.reshape(x)

        return x

class Residual(Sequential):
    def __init__(self, filters, block_id, base_name):
        super(Residual, self).__init__()

        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer=self.k_init,
                           use_bias=False, padding="same", name=f'{base_name}_residual_conv1_{block_id}')
        
        self.activ = LeakyReLU(alpha=0.2, name=f'{base_name}_residual_activ_{block_id}')
        
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer=self.k_init, 
                            use_bias=False, padding="same", name=f'{base_name}_residual_conv2_{block_id}')

        self.add = Add(name=f'{base_name}_residual_add_{block_id}')

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.add([x,residual])

        return x

class Generator(Sequential):
    def __init__(self,  **kwargs):
        super(Generator, self).__init__( **kwargs)

        self.conv0 = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False, 
                            kernel_initializer=self.k_init, name='encoder_conv_0')
        
        self.img_conv = Conv2D(filters=3, kernel_size=3, padding='same', use_bias=False, activation='tanh',
                            kernel_initializer=self.k_init, name='img')
        
        self.mask_conv = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, activation='sigmoid',
                            kernel_initializer=self.k_init, name='mask')
        
        self.enc_dense = Dense(1024, name='latent_dims') 
        self.dec_dense = Dense(4*4*1024, name='dec_dense')
        
        self.conv1 = SeperableConvBlock(64,   1, 'enc')
        self.conv2 = SeperableConvBlock(128,  2, 'enc')
        self.conv3 = SeperableConvBlock(256,  3, 'enc')
        self.conv4 = SeperableConvBlock(512,  4, 'enc')
        self.conv5 = SeperableConvBlock(1024, 5, 'enc')

        self.upscale1 = UpScale(1024*2, 1, 'dec')
        self.upscale2 = UpScale(512*2,  2, 'dec')
        self.upscale3 = UpScale(256*2,  3, 'dec')
        self.upscale4 = UpScale(128*2,  4, 'dec')
        self.upscale5 = UpScale(64*2,   5, 'dec')
        self.upscale6 = UpScale(32*2,   6, 'dec')

        self.res1 = Residual(512, 1, 'dec')
        self.res2 = Residual(256, 2, 'dec')
        self.res3 = Residual(128, 3, 'dec')
        self.res4 = Residual(64,  4, 'dec')
        self.res5 = Residual(32,  5, 'dec')
        self.res6 = Residual(16,  6, 'dec')
    
    @tf.function
    def call(self, inputs):
        x, resnet_vec = inputs
        #encoder
        x = self.conv0(x) #256
        x = self.conv1(x) #128
        x = self.conv2(x) #64
        x = self.conv3(x) #32
        x = self.conv4(x) #16
        x = self.conv5(x) #8

        x = self.enc_dense(Flatten()(x))
        x = Concatenate()([x, resnet_vec])

        x = self.dec_dense(x)
        x = self.upscale1(Reshape((4,4,1024))(x)) #1024
        x = self.res1(x)
        x = self.upscale2(x)  #512
        x = self.res2(x)
        x = self.upscale3(x) #128
        x = self.res3(x)
        x = self.upscale4(x) #64
        x = self.res4(x)
        x = self.upscale5(x) #32
        x = self.res5(x)
        x = self.upscale6(x) #16
        x = self.res6(x)

        img = self.img_conv(x)
        mask = self.mask_conv(x)

        return img, mask

class Discriminator(Sequential):
    def __init__(self,  **kwargs):
        super(Discriminator, self).__init__( **kwargs)
        
        self.conv1 = ConvBlock(128,  1, 'dis', norm=False)
        
        self.conv2 = ConvBlock(256,  2, 'dis')

        self.conv3 = ConvBlock(512, 3, 'dis')

        self.conv4 = ConvBlock(1024, 4, 'dis')

        self.conv5 = Conv2D(filters=1, kernel_size=16, strides=1, use_bias=False, kernel_initializer=self.k_init,
                            name='dis_conv5', activation='sigmoid')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv1(x)#128
        x = self.conv2(x)#64
        x = self.conv3(x)#32
        x = self.conv4(x)#16
        x = self.conv5(x)#8

        return Reshape((1,))(x)

class AdvserialAutoEncoder():
    def __init__(self, generator, discriminator, loss, gen_optim, disc_optim, alpha, lamda):
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.loss = loss

        self.c = alpha
        self.lamda = lamda

        self.generator_optimizer = gen_optim
        self.discriminator_optimizer = disc_optim


    @tf.function
    def train_step(self, dataset_inputs):
        source, target, real, resnet_vec, lam = dataset_inputs
        with tf.GradientTape() as disc_tape:
            raw, mask = self.generator([source, resnet_vec], training=False)
            masked = mask*raw + (1-mask)*source
          
            x = lam*real + (1-lam)*masked
            y = lam[:, 0, 0, 0]

            yhat = self.discriminator(x, training=True)
            real_disc_loss = self.loss.mse(y, yhat)

        gradients_of_discriminator = disc_tape.gradient(real_disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            raw, mask = self.generator([source, resnet_vec], training=True)
            masked = mask*raw + (1-mask)*source
            
            x = lam*real + (1-lam)*masked
            y = (1-lam[:, 0, 0, 0])

            yhat = self.discriminator(x, training=False)

            fake_disc_loss = self.loss.mse(y, yhat)

            Lr_raw = self.loss.recosntruction_loss(real, raw)
            Lr_masked = self.loss.recosntruction_loss(real, masked)

            Lp_raw = self.loss.perceptual_loss(real, source, raw, self.lamda)
            Lp_masked = self.loss.perceptual_loss(real, source, masked, self.lamda)

            Lgx_raw, Lgy_raw = self.loss.gradinet_loss(real, raw)
            Lgx_masked, Lgy_masked = self.loss.gradinet_loss(real, masked)

            Lm, Lm_x, Lm_y = self.loss.m_loss(mask)

            total_loss = (self.c[0]*fake_disc_loss +
                          self.c[1]*Lr_raw  + self.c[1]*Lr_masked + 
                          self.c[2]*Lgx_raw + self.c[2]*Lgy_raw   + self.c[2]*Lgx_masked+self.c[2]*Lgy_masked + 
                          self.c[3]*Lp_raw  + self.c[3]*Lp_masked + 
                          self.c[4]*Lm + 
                          self.c[5]*Lm_x + self.c[5]*Lm_y)

        gradients_of_generator = tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

       
        return [raw, mask, masked, tf.convert_to_tensor((real_disc_loss, fake_disc_loss,
                                    Lr_raw, Lr_masked, Lp_raw,Lp_masked, 
                                    Lgx_raw+Lgy_raw, Lgy_masked+Lgx_masked, Lm+Lm_x+Lm_y))]
            
    
    @tf.function
    def distributed_train_step(self, strategy, dataset_inputs):
        raw, mask, masked, per_replica_losses = strategy.run(self.train_step, args=(dataset_inputs, ))
        return raw, mask, masked, strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,axis=1)