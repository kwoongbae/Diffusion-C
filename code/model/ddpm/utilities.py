import math
import numpy as np
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """
        _extract 설명
        : a에서 index t에 해당하는 값들을 추출하는 함수
        : a를 출력하면 0.01부터 점점 커져 0.9에 이를 때까지 1000개의 숫자가 있는데, 이는 노이즈값들이다.
        """
        
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        
        batch_size = x_shape[0]

        out = tf.gather(a, t)
        # a라는 텐서에서 t라는 인덱스에 해당하는 것들을 out으로 추출한다.
        # a가 뭔지는 모르겠지만 아마 timestep별로의 이미지일듯.
        # 거기서 해당 t에 일치하는 이미지를 추출한다?
        #print("a : {}".format(a))
        #print("a.shape : {}".format(a.shape))
        
        
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """
        q_mean_variance 설명
        : 해당하는 타임스텝에서 노이즈의 평균과 분산을 구하는 함수.
        : 평균은 sqrt(alpha_cumprod) * x_start
        : 분산은 1-alpha_cumprod
        """
        
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """

        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        # beta_t : 노이즈
        # alphas = 1.0 - betas
        # alphas_cumprod = np.cumprod(alphas, axis=0)
        # 0.99부터 시작한다.
        # 즉 1-beta가 0.9, beta는 0.1
        # mean은 현 시점 timestep에 해당하는 노이즈 값을 골라서 x_start에 곱해준다.
        
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        # 여기서 self.alphas_cumprod는 diffusion 과정에서 넣어준 노이즈의 누적합임.
        
        
        #print("self.sqrt_alphas_cumprod : {}".format(self.sqrt_alphas_cumprod))
        #print("mean, value : {},{}".format(mean, value))
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """
        **q_sample 설명**
        : 원본 이미지에 노이즈를 점차 입히는 함수
        : x_0 * alpha_cumprod + sqrt(1-alpha_cumprod)*noise = noise입힌 이미지.
        """
        
        #print("x_start : {}".format(tf.shape(x_start)))
        # 이거는 맨 처음에만 뜨고 안뜸.
        # (4,)이렇게 shape가 뜬다.왜 4지?
        #print("t : {}".format(t))
        #print("noise : {}".format(tf.shape(noise)))
        
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """

        x_start_shape = tf.shape(x_start)
        result_x = self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start 
        result_noise = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)* noise
        result = result_x + result_noise

        return result
        # result는 (1,64,64,3)의 shape을 가지고 있으며, 
        # q_sample()는 한 개의 이미지에 노이즈를 넣어주는 과정임.
        # 그러면 배치사이즈가 128이니깐, q_sample()을 128번 하는건가..?

    def predict_start_from_noise(self, x_t, t, noise):
        """
        predict_start_from_noise 설명
        : sqrt(1/alpha_cumprod) * x_t - sqrt(1/alpha_cumprod-1)*noise
        """
        
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        import matplotlib.pyplot as plt
        #plt.imshow
        #print("x.shape : {}".format(x.shape))
        #print("before x : {}".format(x))
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        #plt.imshow(x_recon[0])
        #plt.show()
        #print("after x_recon : {}".format(x_recon))
        #print("x_recon.shape : {}".format(x_recon.shape))
        # x, x_recon 모두 1000개이다.
        # 각각의 shape 은 (16,64,64,3)
        # 아마 x-x_recon을 통해 노이즈를 추정하려고 하는듯.
        
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        # 그리고 새롭게 나온 x_recon을 넣어 다시 평균과 기타 등등을 구한다. 
        
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffuison model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
        # 이 return 값이 바로 중간중간에 plot되는 이미지들..
    
    def timestep_diffuse(self, sample):
        fig = plt.figure(figsize=(15,30))
        
        images = sample
        
        print(tf.shape(images))
        images_expanded = tf.expand_dims(images,0)
#         plt.imshow(images_expanded[0])
#         plt.show()
        print("{}- max:{}-min:{}".format(images_expanded.shape, np.max(images_expanded), np.min(images_expanded)))
        #print(tf.shape(images_expanded))
        noise = tf.random.normal(shape= tf.shape(images_expanded), dtype=images.dtype)
        # print(tf.shape(images))
        # (64,64,3)
        
        for index, i in enumerate([0, 50,100,150,200,300,500,999]):
            noisy_im = self.q_sample(images_expanded, np.array([i,]), noise)
            #print("noisy_im shape")
            #print(noisy_im.shape)
            plt.subplot(1, 8, index+1)
            plt.imshow(tf.squeeze(noisy_im))
            plt.axis('off')
        plt.show()