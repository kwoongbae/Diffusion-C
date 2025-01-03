import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

# WARNING 이상의 로그 메시지만 출력되도록 설정
logging.basicConfig(level=logging.WARNING)
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from scipy import linalg

def resize_and_crop_image(image):
    # 이미지 크기를 조정합니다.
    image = tf.image.resize(image, size=(299, 299), method=tf.image.ResizeMethod.BILINEAR)
    
    # 이미지를 중앙을 기준으로 crop합니다.
#     image = tf.image.central_crop(image, central_fraction=0.875)
    
    # VGG16 모델의 입력에 맞게 전처리합니다.
    image = preprocess_input(image)
    
    return image.numpy()

def calculate_fid(images1, images2, batch_size=64):  
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    if images1.shape[-1] !=3:
        images1 = np.tile(images1,3)
    if images2.shape[-1] !=3:
        images2 = np.tile(images2,3)

    images1 = np.array([resize_and_crop_image(image) for image in images1])
    images2 = np.array([resize_and_crop_image(image) for image in images2])
    print(images1.shape,images2.shape)
    n1 = images1.shape[0]
    n2 = images2.shape[0]

    pred_arr1 = np.empty((n1, 2048))
    pred_arr2 = np.empty((n2, 2048))

    for i in range(0, n1, batch_size):
        batch = images1[i:i + batch_size]
        pred = model.predict(batch)
        pred_arr1[i:i + batch_size] = pred

    for i in range(0, n2, batch_size):
        batch = images2[i:i + batch_size]
        pred = model.predict(batch)
        pred_arr2[i:i + batch_size] = pred

    mu1, sigma1 = np.mean(pred_arr1, axis=0), np.cov(pred_arr1, rowvar=False)
    mu2, sigma2 = np.mean(pred_arr2, axis=0), np.cov(pred_arr2, rowvar=False)
    sigma1 = np.matrix(sigma1)
    sigma2 = np.matrix(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid



#===
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from scipy import linalg


# def resize_and_crop_image(image):
#     # 이미지 크기를 조정합니다.
#     image = tf.image.resize(image, size=(299, 299), method=tf.image.ResizeMethod.BILINEAR)
    
#     # 이미지를 중앙을 기준으로 crop합니다.
# #     image = tf.image.central_crop(image, central_fraction=0.875)
    
#     # VGG16 모델의 입력에 맞게 전처리합니다.
#     image = preprocess_input(image)
    
#     return image.numpy()

# def calculate_fid_score(images1, images2):
#     # Inception 모델을 불러옵니다.
#     inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    
#     images1 = np.array([resize_and_crop_image(image) for image in images1])
#     images2 = np.array([resize_and_crop_image(image) for image in images2])

#     # 이미지를 Inception 모델의 입력에 맞게 전처리합니다.
#     images1 = preprocess_input(images1)
#     images2 = preprocess_input(images2)
    
#     # 두 데이터셋의 특징을 추출합니다.
#     features1 = inception_model.predict(images1)
#     features2 = inception_model.predict(images2)

#     # 각 데이터셋의 평균과 공분산을 계산합니다.
#     mean1, cov1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
#     mean2, cov2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

#     # 두 데이터셋 간의 평균 차이와 공분산 차이를 계산합니다.
#     diff_mean = mean1 - mean2
#     diff_cov = linalg.sqrtm(cov1.dot(cov2))

#     # Fréchet distance를 계산합니다.
#     fid = np.real(diff_mean.dot(diff_mean) + np.trace(cov1 + cov2 - 2*diff_cov))
    
#     return fid