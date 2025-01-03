"""
Title: DCGAN to generate face images
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/04/29
Last modified: 2021/01/01
Description: A simple DCGAN trained using `fit()` by overriding `train_step` on CelebA images.
Accelerator: GPU
"""
"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

def dcgan_train(dataset_name, noise_name, epochs, ):
  train_x = np.load("./original_datasets/{}/{}.npy".format(dataset_name, noise_name))
  train_x = tf.image.resize(train_x, (64,64)).numpy()
  train_x = train_x.astype("float32")
  train_images = (train_x - 127.5) / 127.5
  print("####################################################")
  
  print("{} with {} dataset loading is completed".format(dataset_name, noise_name))
  print("####################################################")
  print()

  IMG_SHAPE = (64, 64, 3)
  BATCH_SIZE = 128
  noise_dim = 128
  epochs = epochs
  """
  ## Create the discriminator

  It maps a 64x64 image to a binary classification score.
  """
  

  discriminator = keras.Sequential(
      [
          keras.Input(shape=(64, 64, 3)),
          
          layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.Dropout(0.25),
          
          layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.Dropout(0.25),
          
          layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.Dropout(0.25),
          
          layers.Conv2D(512, kernel_size=4, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.Dropout(0.25),
          
          layers.Flatten(),
          layers.Dense(256),
          layers.Dropout(0.25),
          layers.Dense(1, activation="sigmoid"),
      ],
      name="discriminator",
  )

  """
  ## Create the generator

  It mirrors the discriminator, replacing `Conv2D` layers with `Conv2DTranspose` layers.
  """

  latent_dim = 128

  generator = keras.Sequential(
      [
          keras.Input(shape=(latent_dim,)),
          layers.Dense(4 * 4 * 1024),
          layers.Reshape((4, 4, 1024)),
          
          layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"),
      ],
      name="generator",
  )

  """
  ## Override `train_step`
  """


  class GAN(keras.Model):
      def __init__(self, discriminator, generator, latent_dim):
          super().__init__()
          self.discriminator = discriminator
          self.generator = generator
          self.latent_dim = latent_dim

      def compile(self, d_optimizer, g_optimizer, loss_fn):
          super().compile()
          self.d_optimizer = d_optimizer
          self.g_optimizer = g_optimizer
          self.loss_fn = loss_fn
          self.d_loss_metric = keras.metrics.Mean(name="d_loss")
          self.g_loss_metric = keras.metrics.Mean(name="g_loss")

      @property
      def metrics(self):
          return [self.d_loss_metric, self.g_loss_metric]

#       def save_images(self):
#         random_latent_vectors = tf.random.normal(shape=(64, self.latent_dim))
#         generated_images = self.generator(random_latent_vectors)
#         generated_images = (generated_images * 127.5) + 127.5
#         generated_images = generated_images.numpy()
#         img_dir = "./imgs/GAN/{}".format(dataset_name)
#         if not os.path.exists(img_dir):
#           os.makedirs(img_dir)
#         for i in range(len(generated_images)):
#           img = keras.preprocessing.image.array_to_img(generated_images[i])
#           img.save("{}/{}_image_{}.png".format(img_dir, noise_name, i))
#         print()
#         print("-----------------")
#         print("Images of {} with {} saving is completed".format(dataset_name, noise_name))
#         print("-----------------")
#         print()

      def train_step(self, real_images):
          # Sample random points in the latent space
          batch_size = tf.shape(real_images)[0]
          random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

          # Decode them to fake images
          generated_images = self.generator(random_latent_vectors)

          # Combine them with real images
          combined_images = tf.concat([generated_images, real_images], axis=0)

          # Assemble labels discriminating real from fake images
          labels = tf.concat(
              [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
          )
          # Add random noise to the labels - important trick!
          labels += 0.05 * tf.random.uniform(tf.shape(labels))

          # Train the discriminator
          with tf.GradientTape() as tape:
              predictions = self.discriminator(combined_images)
              d_loss = self.loss_fn(labels, predictions)
          grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
          self.d_optimizer.apply_gradients(
              zip(grads, self.discriminator.trainable_weights)
          )

          # Sample random points in the latent space
          random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

          # Assemble labels that say "all real images"
          misleading_labels = tf.zeros((batch_size, 1))

          # Train the generator (note that we should *not* update the weights
          # of the discriminator)!
          with tf.GradientTape() as tape:
              predictions = self.discriminator(self.generator(random_latent_vectors))
              g_loss = self.loss_fn(misleading_labels, predictions)
          grads = tape.gradient(g_loss, self.generator.trainable_weights)
          self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

          # Update metrics
          self.d_loss_metric.update_state(d_loss)
          self.g_loss_metric.update_state(g_loss)
          return {
              "d_loss": self.d_loss_metric.result(),
              "g_loss": self.g_loss_metric.result(),
          }


  """
  ## Create a callback that periodically saves generated images
  """


  class GANMonitor(keras.callbacks.Callback):
      def __init__(self,dataset_name, noise_name,num_img=6, latent_dim=128, ):
          self.dataset_name = dataset_name
          self.noise_name = noise_name
          self.num_img = num_img
          self.latent_dim = latent_dim

      def on_epoch_end(self, epoch, logs=None):
          random_latent_vectors = tf.random.normal(shape=(16, self.latent_dim))
          generated_samples = self.model.generator(random_latent_vectors)
          print()
          print(np.min(generated_samples), np.max(generated_samples), np.mean(generated_samples))
          
          generated_samples = generated_samples.numpy()  
          generated_samples = (generated_samples*127.5) + 127.5
          generated_samples = generated_samples.astype(np.uint8)


          num_rows = 2
          num_cols = 8
          _, ax = plt.subplots(num_rows, num_cols, figsize=(12,5))

          for i, image in enumerate(generated_samples):
            ax[i // num_cols, i % num_cols].imshow(image)
            ax[i // num_cols, i % num_cols].axis("off")
          plt.tight_layout()
          plt.show()

          if epoch == epochs - 1:
            model_dir = "./saved_models/dcgan/{}".format(self.dataset_name)
            if not os.path.exists(model_dir):
              os.makedirs(model_dir)
            self.model.generator.save_weights('{}/weights_of_{}_{}epochs'.format(model_dir, self.noise_name, epochs))
            print("{} with {} dataset trained model saving is completed".format(self.dataset_name, self.noise_name))


  """
  ## Train the end-to-end model
  """
  
  gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
  gan.compile(
      d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
      g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
      loss_fn=keras.losses.BinaryCrossentropy(),
  )

  gan.fit(
      train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[GANMonitor(dataset_name, noise_name, num_img=10, latent_dim=latent_dim)]
  )

  """
  Some of the last generated images around epoch 30
  (results keep improving after that):

  ![results](https://i.imgur.com/h5MtQZ7l.png)
  """

def dcgan_inference(dataset_name, noise_name, epochs, num_imgs):
  latent_dim = 128
  IMG_SHAPE = (64, 64, 3)
  BATCH_SIZE = 128
  noise_dim = 128

  generator = keras.Sequential(
      [
          keras.Input(shape=(latent_dim,)),
          layers.Dense(4 * 4 * 1024),
          layers.Reshape((4, 4, 1024)),
          
          layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding="same"),
          layers.LeakyReLU(alpha=0.2),
          
          layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"),
      ],
      name="generator",
  ) 
  model_dir = "./saved_models/dcgan/{}".format(dataset_name)
  generator.load_weights('{}/weights_of_{}_{}epochs'.format(model_dir, noise_name, epochs))
  random_latent_vectors = tf.random.normal(shape=(num_imgs, 128))
  generated_images = generator(random_latent_vectors)


  return generated_images