import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os

def wgan_train(dataset_name, noise_name, epochs, ):
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

  def conv_block(
      x,
      filters,
      activation,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding="same",
      use_bias=True,
      use_bn=False,
      use_dropout=False,
      drop_value=0.5,
  ):
      x = layers.Conv2D(
          filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
      )(x)
      if use_bn:
          x = layers.BatchNormalization()(x)
      x = activation(x)
      if use_dropout:
          x = layers.Dropout(drop_value)(x)
      return x


  def get_discriminator_model():
      img_input = layers.Input(shape=IMG_SHAPE)
      # Zero pad the input to make the input images size to (32, 32, 1).
      x = layers.ZeroPadding2D((2, 2))(img_input)
      x = conv_block(
          x,
          64,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          use_bias=True,
          activation=layers.LeakyReLU(0.2),
          use_dropout=False,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          128,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=True,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          256,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=True,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          512,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=False,
          drop_value=0.3,
      )

      x = layers.Flatten()(x)
      x = layers.Dropout(0.2)(x)
      x = layers.Dense(1)(x)

      d_model = keras.models.Model(img_input, x, name="discriminator")
      return d_model


  d_model = get_discriminator_model()

  """
  ## Create the generator
  """


  def upsample_block(
      x,
      filters,
      activation,
      kernel_size=(3, 3),
      strides=(1, 1),
      up_size=(2, 2),
      padding="same",
      use_bn=False,
      use_bias=True,
      use_dropout=False,
      drop_value=0.3,
  ):
      x = layers.UpSampling2D(up_size)(x)
      x = layers.Conv2D(
          filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
      )(x)

      if use_bn:
          x = layers.BatchNormalization()(x)

      if activation:
          x = activation(x)
      if use_dropout:
          x = layers.Dropout(drop_value)(x)
      return x


  def get_generator_model():
      noise = layers.Input(shape=(noise_dim,))
      x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(0.2)(x)

      x = layers.Reshape((4, 4, 256))(x)
      x = upsample_block(
          x,
          128,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x,
          64,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x,
          64,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x, 3, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
      )
      # At this point, we have an output which has the same shape as the input, (32, 32, 1).
      # We will use a Cropping2D layer to make it (28, 28, 1).
      # x = layers.Cropping2D((2, 2))(x)

      g_model = keras.models.Model(noise, x, name="generator")
      return g_model


  g_model = get_generator_model()

  """
  ## Create the WGAN-GP model

  Now that we have defined our generator and discriminator, it's time to implement
  the WGAN-GP model. We will also override the `train_step` for training.
  """


  class WGAN(keras.Model):
      def __init__(
          self,
          discriminator,
          generator,
          latent_dim,
          discriminator_extra_steps=3,
          gp_weight=10.0,
      ):
          super().__init__()
          self.discriminator = discriminator
          self.generator = generator
          self.latent_dim = latent_dim
          self.d_steps = discriminator_extra_steps
          self.gp_weight = gp_weight

      def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
          super().compile()
          self.d_optimizer = d_optimizer
          self.g_optimizer = g_optimizer
          self.d_loss_fn = d_loss_fn
          self.g_loss_fn = g_loss_fn

      def gradient_penalty(self, batch_size, real_images, fake_images):
          """Calculates the gradient penalty.

          This loss is calculated on an interpolated image
          and added to the discriminator loss.
          """
          # Get the interpolated image
          alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
          diff = fake_images - real_images
          interpolated = real_images + alpha * diff

          with tf.GradientTape() as gp_tape:
              gp_tape.watch(interpolated)
              # 1. Get the discriminator output for this interpolated image.
              pred = self.discriminator(interpolated, training=True)

          # 2. Calculate the gradients w.r.t to this interpolated image.
          grads = gp_tape.gradient(pred, [interpolated])[0]
          # 3. Calculate the norm of the gradients.
          norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
          gp = tf.reduce_mean((norm - 1.0) ** 2)
          return gp

#       def save_images(self):
#           random_latent_vectors = tf.random.normal(shape=(64, self.latent_dim))
#           generated_images = self.generator(random_latent_vectors)
#           generated_images = (generated_images * 127.5) + 127.5
#           generated_images = generated_images.numpy()
#           img_dir = "./imgs/WGAN/{}".format(dataset_name)
#           if not os.path.exists(img_dir):
#             os.makedirs(img_dir)
#           for i in range(len(generated_images)):
#             img = keras.preprocessing.image.array_to_img(generated_images[i])
#             img.save("{}/{}_image_{}.png".format(img_dir, noise_name, i))
#           print()
#           print("-----------------")
#           print("Images of {} with {} saving is completed".format(dataset_name, noise_name))
#           print("-----------------")
#           print()

      def train_step(self, real_images):
          if isinstance(real_images, tuple):
              real_images = real_images[0]

          batch_size = tf.shape(real_images)[0]

          for i in range(self.d_steps):
              # Get the latent vector
              random_latent_vectors = tf.random.normal(
                  shape=(batch_size, self.latent_dim)
              )
              with tf.GradientTape() as tape:
                  # Generate fake images from the latent vector
                  fake_images = self.generator(random_latent_vectors, training=True)
                  # Get the logits for the fake images
                  fake_logits = self.discriminator(fake_images, training=True)
                  # Get the logits for the real images
                  real_logits = self.discriminator(real_images, training=True)

                  # Calculate the discriminator loss using the fake and real image logits
                  d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                  # Calculate the gradient penalty
                  gp = self.gradient_penalty(batch_size, real_images, fake_images)
                  # Add the gradient penalty to the original discriminator loss
                  d_loss = d_cost + gp * self.gp_weight

              # Get the gradients w.r.t the discriminator loss
              d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
              # Update the weights of the discriminator using the discriminator optimizer
              self.d_optimizer.apply_gradients(
                  zip(d_gradient, self.discriminator.trainable_variables)
              )

          # Train the generator
          # Get the latent vector
          random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
          with tf.GradientTape() as tape:
              # Generate fake images using the generator
              generated_images = self.generator(random_latent_vectors, training=True)
              # Get the discriminator logits for fake images
              gen_img_logits = self.discriminator(generated_images, training=True)
              # Calculate the generator loss
              g_loss = self.g_loss_fn(gen_img_logits)

          # Get the gradients w.r.t the generator loss
          gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
          # Update the weights of the generator using the generator optimizer
          self.g_optimizer.apply_gradients(
              zip(gen_gradient, self.generator.trainable_variables)
          )
          return {"d_loss": d_loss, "g_loss": g_loss}


  """
  ## Create a Keras callback that periodically saves generated images
  """


  class GANMonitor(keras.callbacks.Callback):
      def __init__(self,dataset_name, noise_name,num_img=6, latent_dim=128, ):
          self.dataset_name = dataset_name
          self.noise_name = noise_name
          self.num_img = num_img
          self.latent_dim = latent_dim

      def on_epoch_end(self, epoch, logs=None):
          random_latent_vectors = tf.random.normal(shape=(16, self.latent_dim))
          generated_images = self.model.generator(random_latent_vectors)
          generated_images = (generated_images * 127.5) + 127.5
          generated_samples = tf.clip_by_value(generated_images,0.0,255.0).numpy().astype(np.uint8)

          num_rows = 2
          num_cols = 8
          _, ax = plt.subplots(num_rows, num_cols, figsize=(12,5))

          for i, image in enumerate(generated_samples):
            ax[i // num_cols, i % num_cols].imshow(image)
            ax[i // num_cols, i % num_cols].axis("off")
          plt.tight_layout()
          plt.show()


          if epoch == epochs - 1:
            model_dir = "./saved_models/wgan_gp/{}".format(self.dataset_name)
            if not os.path.exists(model_dir):
              os.makedirs(model_dir)
            self.model.generator.save_weights('{}/weights_of_{}_{}epochs'.format(model_dir, self.noise_name, epochs))
            print("{} with {} dataset trained model saving is completed".format(self.dataset_name, self.noise_name))
            

  # Instantiate the optimizer for both networks
  # (learning_rate=0.0002, beta_1=0.5 are recommended)
  generator_optimizer = keras.optimizers.Adam(
      learning_rate=0.0002, beta_1=0.5, beta_2=0.9
  )
  discriminator_optimizer = keras.optimizers.Adam(
      learning_rate=0.0002, beta_1=0.5, beta_2=0.9
  )

  # Define the loss functions for the discriminator,
  # which should be (fake_loss - real_loss).
  # We will add the gradient penalty later to this loss function.
  def discriminator_loss(real_img, fake_img):
      real_loss = tf.reduce_mean(real_img)
      fake_loss = tf.reduce_mean(fake_img)
      return fake_loss - real_loss


  # Define the loss functions for the generator.
  def generator_loss(fake_img):
      return -tf.reduce_mean(fake_img)


  # Instantiate the customer `GANMonitor` Keras callback.
  cbk = GANMonitor(dataset_name, noise_name, num_img=3, latent_dim=noise_dim)

  # Get the wgan model
  wgan = WGAN(
      discriminator=d_model,
      generator=g_model,
      latent_dim=noise_dim,
      discriminator_extra_steps=3,
  )

  # Compile the wgan model
  wgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
      g_loss_fn=generator_loss,
      d_loss_fn=discriminator_loss,
  )

  # Start training
  wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])

#   wgan.save_images()

def wgan_inference(dataset_name, noise_name, epochs, num_imgs):
  IMG_SHAPE = (64, 64, 3)
  BATCH_SIZE = 128
  noise_dim = 128
  epochs = epochs

  def conv_block(
      x,
      filters,
      activation,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding="same",
      use_bias=True,
      use_bn=False,
      use_dropout=False,
      drop_value=0.5,
  ):
      x = layers.Conv2D(
          filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
      )(x)
      if use_bn:
          x = layers.BatchNormalization()(x)
      x = activation(x)
      if use_dropout:
          x = layers.Dropout(drop_value)(x)
      return x


  def get_discriminator_model():
      img_input = layers.Input(shape=IMG_SHAPE)
      # Zero pad the input to make the input images size to (32, 32, 1).
      x = layers.ZeroPadding2D((2, 2))(img_input)
      x = conv_block(
          x,
          64,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          use_bias=True,
          activation=layers.LeakyReLU(0.2),
          use_dropout=False,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          128,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=True,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          256,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=True,
          drop_value=0.3,
      )
      x = conv_block(
          x,
          512,
          kernel_size=(5, 5),
          strides=(2, 2),
          use_bn=False,
          activation=layers.LeakyReLU(0.2),
          use_bias=True,
          use_dropout=False,
          drop_value=0.3,
      )

      x = layers.Flatten()(x)
      x = layers.Dropout(0.2)(x)
      x = layers.Dense(1)(x)

      d_model = keras.models.Model(img_input, x, name="discriminator")
      return d_model


  d_model = get_discriminator_model()

  """
  ## Create the generator
  """


  def upsample_block(
      x,
      filters,
      activation,
      kernel_size=(3, 3),
      strides=(1, 1),
      up_size=(2, 2),
      padding="same",
      use_bn=False,
      use_bias=True,
      use_dropout=False,
      drop_value=0.3,
  ):
      x = layers.UpSampling2D(up_size)(x)
      x = layers.Conv2D(
          filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
      )(x)

      if use_bn:
          x = layers.BatchNormalization()(x)

      if activation:
          x = activation(x)
      if use_dropout:
          x = layers.Dropout(drop_value)(x)
      return x


  def get_generator_model():
      noise = layers.Input(shape=(noise_dim,))
      x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(0.2)(x)

      x = layers.Reshape((4, 4, 256))(x)
      x = upsample_block(
          x,
          128,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x,
          64,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x,
          64,
          layers.LeakyReLU(0.2),
          strides=(1, 1),
          use_bias=False,
          use_bn=True,
          padding="same",
          use_dropout=False,
      )
      x = upsample_block(
          x, 3, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
      )
      # At this point, we have an output which has the same shape as the input, (32, 32, 1).
      # We will use a Cropping2D layer to make it (28, 28, 1).
      # x = layers.Cropping2D((2, 2))(x)

      g_model = keras.models.Model(noise, x, name="generator")
      return g_model


  g_model = get_generator_model()

  generator = g_model
#   generator.summary()
  generator.load_weights("./saved_models/wgan_gp/{}/weights_of_{}_{}epochs".format(dataset_name,noise_name, epochs))
  random_latent_vectors = tf.random.normal(shape=(num_imgs, 128))
  generated_images = generator(random_latent_vectors)


  return generated_images