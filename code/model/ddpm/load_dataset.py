
import tensorflow as tf
import tensorflow_datasets as tfds

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

## Change the type of the noise
# none, gauss, poisson, s&p, speckle


def resize_and_rescale(img, size):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)

    img = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # Resize
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)
    # Rescale the pixel values
    img = img / 127.5 - 1.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img

