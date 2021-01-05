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
        #mats[..., 2] += tform

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

    def _elastic_transform(self, image, alpha, sigma, alpha_affine=4):
        random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)