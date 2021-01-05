import tensorflow as tf
import numpy as np
import cv2



def write_model_layers(model, layer_names, step, summary_writer):
    with summary_writer.as_default():
        for l in model.submodules:
            if (l.name in layer_names):
                layer_weights = l.weights[0]
                tf.summary.histogram(l.name, layer_weights, step=step)

def write_images(data, shape, step, summary_writer):
    with summary_writer.as_default():
        for name, img in data.items():
            img = np.reshape(img, (-1, shape[0], shape[1], shape[2]))
            tf.summary.image(name, img, step=step)

def write_log(logs, step, summary_writer):
    with summary_writer.as_default():
        for name, value in logs.items():
            tf.summary.scalar(name, value, step=step)


def wma(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def resize(images, target_size=(224,224)):
    return np.array([cv2.resize(img, target_size) for img in images])
  
def get_data_generator(train_sequence):
    def generator():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_sequence, use_multiprocessing=False, shuffle=False)
#        multi_enqueuer.start(workers=10, max_queue_size=10)
        multi_enqueuer.start()

        while True:
            s,t,r,l,des = next(multi_enqueuer.get()) # I have three outputs
            yield s,t,r,l,des
            
    return generator

def get_dist_data_gen(train_sequence, strategy):
    generator = get_data_generator(train_sequence)
    dataset = tf.data.Dataset.from_generator(generator,
                                         output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None]))
                                          )

    return strategy.experimental_distribute_dataset(dataset)
