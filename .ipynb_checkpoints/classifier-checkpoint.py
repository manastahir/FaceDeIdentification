import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from resnet import resnet50_backend

global weight_decay
weight_decay = 1e-4


def Vggface2_ResNet50(input_shape=(256,256,3), nb_classes=8631, weights=None, trainable=False, output_layers=None, name="renet_classifier"):
    input_tensor = layers.Input(shape=(224, 224, 3))
    # inputs are of size 224 x 224 x 3
    x = resnet50_backend(input_tensor)

    # AvgPooling
    x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', name='dim_proj')(x)

    y = layers.Dense(nb_classes, activation='softmax',
                            use_bias=False, trainable=True,
                            kernel_initializer='orthogonal',
                            kernel_regularizer=l2(weight_decay),
                            name='classifier_low_dim')(x)

    # Compile
    model = Model(inputs=input_tensor, outputs=y, name=name)
    
    # for idx,l in enumerate(model.layers):
    #     print(idx, l.name, l.output_shape)
        
    if(weights is not None):
        model.load_weights(weights)
    
    if(output_layers is not None):
        model.outputs = [model.layers[l].output for l in output_layers]
    
    model = Model(inputs=input_tensor, outputs=model.outputs, name=name)
    model.compile('adam', 'mse')
    
    input_tensor = layers.Input(shape=input_shape)
    rescaled_input_tensor = input_tensor*255
    resized_input_tensor = tf.image.resize(rescaled_input_tensor, (224, 224))

    y = model(resized_input_tensor)
    
    model = Model(inputs=input_tensor, outputs=y, name=name)
    model.trainable=False
    model.compile('adam', 'mse')

    return model