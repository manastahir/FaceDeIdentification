import tensorflow as tf
from tensorflow.keras import backend as K

class LossFunction():
    def __init__(self, classifier):        
        self.classifier = classifier

    def mse(self, y_true, y_pred):
        return tf.keras.losses.mse(y_true, y_pred)

    def gradients(self, tensor):
        return tf.image.image_gradients(tensor)

    def l1(self, a, b):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(a) == 4:
            return K.mean(K.abs(a - b), axis=[1,2,3])
        elif K.ndim(a) == 2:
            return K.mean(K.abs(a - b), axis=[1])
        else:
            raise NotImplementedError("Calculating L1 loss on tensors that are not 4D or 2D? should not occur for this network")
       
    def perceptual_loss(self, real, target, output, lamda):

        z_features = self.classifier(output)
        t_features = self.classifier(target)
        r_features = self.classifier(real)

        total_perceptual_loss = 0
        
        total_perceptual_loss += (self.l1(r_features[0], z_features[0]))/(64*112*112)
        total_perceptual_loss += (self.l1(r_features[1], z_features[1]))/(256*55*55)
        total_perceptual_loss += (self.l1(r_features[2], z_features[2]))/(512*28*28)
    
        total_perceptual_loss += (self.l1(t_features[3], z_features[3]))/(2048*7*7)
        total_perceptual_loss += (-lamda*(self.l1(t_features[4], z_features[4]))/1)
        
        return total_perceptual_loss    

    def recosntruction_loss(self, x, z):
        return self.l1(z, x)
    
    def gradinet_loss(self, x, z):
        z_dx, z_dy = self.gradients(z)
        x_dx, x_dy = self.gradients(x)

        return self.l1(z_dx, x_dx), self.l1(z_dy, x_dy)

    def m_loss(self, m):
        m_dx, m_dy = self.gradients(m)
        m1 = K.mean(K.abs(m), axis=[1,2,3])
        m2 = K.mean(K.abs(m_dx), axis=[1,2,3])
        m3 = K.mean(K.abs(m_dy), axis=[1,2,3])

        return m1, m2, m3
