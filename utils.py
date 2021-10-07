import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import cv2


class ImageData:

    def __init__(self, img_size, channels, augment_flag=False):
        self.img_size = img_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)#将图像使用JPEG的格式解码从而得到图像对应的三维矩阵。解码之后的结果为一个张量。
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag:
            if self.img_size < 256:
                augment_size = 256
            else:
                augment_size = self.img_size + 30
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def load_test_data(image_path, size=256):
    # img = misc.imread(image_path, mode='RGB')
    # img = misc.imresize(img, [size, size])
    print('image_path: ', image_path)
    img = cv2.imread(image_path, 0)#cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，参数可以直接写0。
    Shape = img.shape
    h = Shape[0] // 4 * 4
    w = Shape[1] // 4 * 4
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = preprocessing(img)

    return img, h, w

#归一化
def preprocessing(x):
    #x = x / 127.5 - 1  # -1 ~ 1
    x = x / 256 # 0 ~ 1
    return x


def augmentation(image, aug_img_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_test_images(images, size, image_path):
    return imsave_test(test_inverse_transform(images), size, image_path)

def test_inverse_transform(images):
    return (images + 0.0) * 255.0

def inverse_transform(images):
    return (images + 1.0) * 255.0

def imsave_test(images, size, path):
    print(path)
    images = np.squeeze(images)
    return cv2.imwrite(path, images)

def imsave(images, size, path):
    print(path)
    return cv2.imwrite(path, merge(images, size))
    # return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')

def gradient(input):
    filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
    filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
    Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
    Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
    return Gradient

def get_weight(content_A, content_B):
    content_A_grad = gradient(content_A)
    content_B_grad = gradient(content_B)
    weight = content_A_grad - content_B_grad
    # threshold = 0.02 * tf.ones_like(weight)
    one = tf.ones_like(weight)
    zero = tf.zeros_like(weight)
    weight = tf.where(weight > 0.02, x=one, y=zero)
    return weight
def blur_weight(weight, blur_size=3):
    mean_filter = tf.ones((blur_size, blur_size, 1, 1), dtype=tf.float32) / (blur_size * blur_size)
    weight_blur = tf.nn.conv2d(weight, filter=mean_filter, strides=[1, 1, 1, 1], padding='SAME')
    return weight_blur
def expand_weight(weight, channel):
    filter = tf.ones((1, 1, 1, channel), dtype=tf.float32)
    weight = tf.nn.conv2d(weight, filter=filter, strides=[1, 1, 1, 1], padding='VALID')
    return weight

def fusion_content(content_A, content_B):
    channel = content_A.get_shape().as_list()[-1]
    for convi in range(channel):
        content_A_convi = tf.expand_dims(content_A[:, :, :, convi], -1)
        content_B_convi = tf.expand_dims(content_B[:, :, :, convi], -1)
        weight_ = get_weight(content_A_convi, content_B_convi)
        if convi == 0:
            weight = weight_
        else:
            weight = weight + weight_
    threshold = channel / 3
    one = tf.ones_like(weight)
    zero = tf.zeros_like(weight)
    weight = tf.where(weight > threshold, x=one, y=zero)
    weight = blur_weight(weight)
    # print("weight shape: ", weight.get_shape().as_list())
    weight = expand_weight(weight, channel)
    fusion_content = weight * content_A + (1 - weight) * content_B
    return fusion_content





