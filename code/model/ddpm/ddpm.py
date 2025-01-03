import math
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
import tensorflow_datasets as tfds
import keras

from . import diffusion
from . import network
from . import utilities

from .diffusion import DiffusionModel
from .network import build_model
from .utilities import GaussianDiffusion

# from network import build_model
# from utilities import GaussianDiffusion
# from diffusion import DiffusionModel

from . import load_dataset
# from load_dataset import resize_and_rescale
# from load_dataset import *
# from corruptions import *
# import noisy
# from noisy import noisy


batch_size = 64
num_epochs = 50  # Just for the sake of demonstration
total_timesteps = 1000
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4
clip_min = -1.0
clip_max = 1.0

img_size = 64
img_channels = 3

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

base_dir = './original_datasets'

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

def train_preprocessing(img):
    return resize_and_rescale(img, size=(img_size, img_size))
    
def ddpm_train(dataset_name, noise_name):
    
    if noise_name == "fog_severity_1":
        train_x = np.load("./original_datasets/severity_exp/{}/{}.npy".format(dataset_name,noise_name))
    elif noise_name == "fog_severity_2":
        train_x = np.load("./original_datasets/severity_exp/{}/{}.npy".format(dataset_name,noise_name))
    elif noise_name == "fog_severity_3":
        train_x = np.load("./original_datasets/severity_exp/{}/{}.npy".format(dataset_name,noise_name))
    elif noise_name == "fog_severity_4":
        train_x = np.load("./original_datasets/severity_exp/{}/{}.npy".format(dataset_name,noise_name))
    elif noise_name == "fog_severity_5":
        train_x = np.load("./original_datasets/severity_exp/{}/{}.npy".format(dataset_name,noise_name))
    else:
        train_x = np.load("./original_datasets/{}/{}.npy".format(dataset_name, noise_name))
#     print("H : {},{}".format(np.min(train_x), np.max(train_x)))
#     original = train_x.astype(np.float64)/127.5-1
#     noise = np.random.normal(size=train_x.shape)
#     t = [250] * train_x.shape[0]
#     train_x = q_sample(original,t,noise)
#     train_x = (train_x+1)*127.5
    
    dataset = tf.data.Dataset.from_tensor_slices((
        train_x
    ))
    dataset1 = dataset.map(train_preprocessing, tf.data.AUTOTUNE)
    dataset1 = dataset1.batch(batch_size).shuffle(batch_size*2).prefetch(tf.data.AUTOTUNE)
    dataset1 = dataset1.repeat(100)
    train_ds = tfds.as_numpy(dataset1)
    
    print("{}-{} datasets loading completed".format(dataset_name, noise_name))

    # Build the unet model
    network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )
    ema_network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )
    ema_network.set_weights(network.get_weights())  # Initially the weights are the same

    # Get an instance of the Gaussian Diffusion utilities
    gdf_util = GaussianDiffusion(timesteps=total_timesteps)

    # Get the model
    model = DiffusionModel(
        network=network,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=total_timesteps,
    )

    
    # Compile the model
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )    
    
    sample = next(iter(train_ds))[0]
#     print(sample.shape)
    print(np.min(sample), np.max(sample))
#     plt.imshow(sample)
    print(np.max(sample),np.min(sample))

    a = gdf_util.timestep_diffuse(sample)     # 노이즈 넣어주는거 시각화하는 코드
    
    #========================5. Model train 및 가중치 저장=================================
    steps_per_epoch = train_x.shape[0]//batch_size-1
    
    print("Model training Starts")
    print()
    
    # Train the model
    if dataset_name != "imagenet11":
        model.fit(
            iter(train_ds),
            batch_size = batch_size,
            epochs= num_epochs,
            steps_per_epoch =steps_per_epoch,
            callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],
        )   
    else:
        model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs = num_epochs)
    
    model.save_weights('./saved_models/ddpm/{}/{}_with_<{}>_with_{}epochs'.format(dataset_name,dataset_name, noise_name, num_epochs))
    new_model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
    )
    new_model.load_weights('./saved_models/ddpm/{}/{}_with_<{}>_with_{}epochs'.format(dataset_name,dataset_name, noise_name, num_epochs))
    
    # ========================6. images save
    generated_images = new_model.generate_images(64)
    path = './img/{}/{}/'.format(dataset_name, noise_name)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(generated_images)):
        plt.imshow(generated_images[i])
        plt.axis('off')
        plt.savefig('img/{}/{}/{}_{}_{}_image{}.png'.format(dataset_name, noise_name, dataset_name, noise_name, num_epochs,i))
    
    print("{}-{} finished".format(dataset_name, noise_name))
    print("========================================================================")
    print()
    
def inference(dataset_name, noise_name, num_imgs):
    gdf_util = GaussianDiffusion(timesteps=total_timesteps)
    network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )
    ema_network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )
    ema_network.set_weights(network.get_weights())
    
    new_model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
    )
    new_model.load_weights('./saved_models/ddpm/{}/{}_with_<{}>_with_{}epochs'.format(dataset_name,dataset_name, noise_name, num_epochs))
    
#     print("Started Inferencing")
    generated_images = new_model.generate_images(num_imgs)
    print(generated_images.shape)
    return generated_images