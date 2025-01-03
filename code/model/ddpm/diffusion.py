import math
import numpy as np
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=int(self.timesteps), shape=(batch_size,), dtype=tf.int64
        )
        # 이렇게 되면 t는 0부터 1000까지의 정수가 있는 list가 될 것임.
        # >>> Q : 근데 왜 random으로 지정하는걸까?
        # 
        # i=0
        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            mean_value = tf.reduce_mean(noise)
            #print("mean is {}".format(mean_value))
            # image와 같은 크기의 노이즈를 추가.
            # 이 노이즈를 이미지(x_T, x_t)에 추가해줄 예정.

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            # i+=1
            # print(i)
            # 이것도 맨 처음에만 뜨고 안뜸.
            
            # q_sample()함수를 통해 이미지에 노이즈를 입혀가기 시작함.
            # q_sample()에 images들이랑, 0에서 1000까지의 timestep이랑, images와 같은 shape의 노이즈를 넣어줌.
            # 찍어보니깐 images는 -1로 도배되어있고, t는 [0]이며, noise는 말그래도 노이즈인듯.
            # images와 noise모두 (1,64,64,3)의 shape을 가지고 있음.
            
            # 그러면 q_sample이 알아서 결과물을 주나 봄
            # Q : 결과물은 무엇일까?
            #print("images_t : {}".format(tf.shape(images_t)))
            #print("images_t : {}".format(images_t))
            
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)
            # >>> Q : Epoch1 이거는 어디서 뜨는 걸까. 
            #### fit하면 자연스럽게 나오게 됨.
            
            # >>> Q : train_step과 ddpm패키지와의 연결고리는 어디일까.

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)
            # 이거는 keras.Model 상속받으면서 얻게되는 loss 계산 함수인듯.
            

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def generate_images(self, num_images=16):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, img_size, img_size, img_channels), dtype=tf.float32
        )
        # 일단은 위와 같은 shape을 가지는 난수 데이터셋을 생성
        # 이 함수는 중간중간 plot을 띄우기 위해 16개로 num_images를 설정함.

        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            if t %50 == 0:
                print(t)
            # 이거는 학습 epoch마다 계속 뜸.
            
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            # 아마 16개의 이미지마다 timestep index를 달아주기 위해 사용.
            #print("tt : {}".format(tt[0]))
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            #print("pred_noise.shape : {}".format(pred_noise.shape))
            # 내 예상으로는 (16,64,64,3)
#             if t%50==0:
#                 img = pred_noise[0]
#                 print(np.mean(img))
#                 fig = plt.figure(figsize=(4, 2.6))
#                 plt.ylim(0,150)
#                 plt.hist(img.ravel(), bins=256, range=(-1, 1), fc="{}".format("blue"), ec="{}".format("gray"))
#                 plt.ylabel('Frequency')
#                 #plt.imshow()
#                 plt.show()
            
            #print("pred_noise : {}".format(np.mean(pred_noise)))
            # 음 그러면 ema_network는 매 epoch마다 훈련이 되고,,
            # 가중치가 업데이트 된 ema_network에 샘플을 넣어서 noise를 예측.
            # pred_noise는 넣어줬던 노이즈의 양을 예측하는 것 아닌가.
            # >>> Q. ema_network도 그냥 network와 똑같은거 아닌가. 왜 그냥 sample이 아닌, [samples,tt]를 넣는가?
            #### tt를 출력해보니, timestep을 1000부터 거꾸로 갈 때마다 생성되는 16개의 이미지한테 time을 라벨링해주기 위해 사용하는 것이다.
            
            #print("tt:{}".format(tt))
            #print("[samples,tt] : {}".format(samples,tt))
            
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            #print("samples : {},{}".format(np.mean(samples), np.var(samples)))
            # p_sample()에 예측한 노이즈와 난수 노이즈, tt를 넣어준다.
            # 그러면 총 16개의 이미지가 나오게 된다.
        
#             print(t)
        # 3. Return generated samples
        return samples

    def plot_images(
        self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
            .numpy()
            .astype(np.uint8)
        )

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()