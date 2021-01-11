import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
import elasticdeform as ed

class ImageAugmentation():
    def __init__(self, params):
      self._image_size = params["source_size"]
      self.batch_size = params["batch_size"]
      self._rotation_range = params["rotation_range"]
      self._zoom_range = params["zoom_amount"] /100
      self._shift_range = params["shift_range"] /100
      self._flip = params["flip"] /100

      self._alpha = params["alpha"]
      self._sigma = params["sigma"]

      np.random.seed(params["random_state"])

    def trasnform(self, batch):
        rotation = np.random.uniform(-self._rotation_range,
                                      self._rotation_range,
                                      size=self.batch_size).astype("float32")

        scale = np.random.uniform(1 - self._zoom_range,
                                  1 + self._zoom_range,
                                  size=self.batch_size).astype("float32")

        tform = np.random.uniform(
            -self._shift_range,
            self._shift_range,
            size=(self.batch_size, 2)).astype("float32") * self._image_size

        mats = np.array(
            [cv2.getRotationMatrix2D((self._image_size // 2, self._image_size // 2),
                                      rot,
                                      scl)
              for rot, scl in zip(rotation, scale)]).astype("float32")

        batch = np.array([cv2.warpAffine(image, mat,
                              (self._image_size, self._image_size),
                              borderMode=cv2.BORDER_REPLICATE)
                              for image, mat in zip(batch, mats)])
      
        batch = np.array([ed.deform_random_grid(image, sigma=self._sigma, points=self._alpha, axis=(0, 1), mode='reflect')
                        for image in batch])
        
        return batch


    def random_flip(self, batch):
        randoms = np.random.rand(self.batch_size)
        indices = np.where(randoms >  self._flip)[0]
        batch[indices] = batch[indices, :, ::-1]
        
        return batch