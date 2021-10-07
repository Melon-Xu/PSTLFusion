import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad > 0:
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')#映射填充，上下（1维）填充顺序和paddings是相反的，左右（零维）顺序补齐

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            w = weights_spectral_norm(w)#对权重进行处理，可加速收敛
            x = tf.nn.conv2d(input=x, filter=w, strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=True, scope='deconv'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[1] * stride, x_shape[2] * stride]
        else:
            output_shape = [x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0)]
        up_sample = tf.image.resize_images(x, output_shape, method=1)#调整图像大小，method=1代表最近邻居里插值方式
        if sn:
            x = conv(up_sample, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        # if sn:
        #     w = tf.get_variable("kernel", shape=[kernel, kernel, up_sample.get_shape()[-1], channels], initializer=weight_init,
        #                         regularizer=weight_regularizer)
        #     x = tf.nn.conv2d(up_sample, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
        #
        #     if use_bias:
        #         bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        #         x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x


# def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv'):
#     with tf.variable_scope(scope):
#         x_shape = x.get_shape().as_list()
#
#         if padding == 'SAME':
#             output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
#
#         else:
#             output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
#                             x_shape[2] * stride + max(kernel - stride, 0), channels]
#
#         if sn:
#             w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
#                                 regularizer=weight_regularizer)
#             x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
#                                        strides=[1, stride, stride, 1], padding=padding)
#
#             if use_bias:
#                 bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
#                 x = tf.nn.bias_add(x, bias)
#
#         else:
#             x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
#                                            kernel_size=kernel, kernel_initializer=weight_init,
#                                            kernel_regularizer=weight_regularizer,
#                                            strides=stride, padding=padding, use_bias=use_bias)
#
#         return x


def attribute_connet(x, channels, use_bias=True, sn=True, scope='attribute'):
    with tf.variable_scope(scope):
        x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)#全连接层，units为输出的通道数
        return x


def fully_conneted(x, channels, use_bias=True, sn=True, scope='fully'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        x_channel = shape[-1]
        if sn:
            w = tf.get_variable("kernel", [x_channel, channels], tf.float32, initializer=weight_init,
                                regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)
            print('fully_connected shape: ', x.get_shape().as_list())
        return x


def gaussian_noise_layer(x, is_training=False):
    if is_training:
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
        return x + noise

    else:
        return x


##################################################################################
# Block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = layer_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = layer_norm(x)

        return x + x_init


def basic_block(x_init, channels, use_bias=True, sn=True, scope='basic_block'):
    with tf.variable_scope(scope):
        x = lrelu(x_init, 0.2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        x = lrelu(x, 0.2)
        x = conv_avg(x, channels, use_bias=use_bias, sn=sn)

        shortcut = avg_conv(x_init, channels, use_bias=use_bias, sn=sn)

        return x + shortcut


def mis_resblock(x_init, z, channels, use_bias=True, sn=True, scope='mis_resblock'):
    with tf.variable_scope(scope):
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1])  # expand

        with tf.variable_scope('mis1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
                     scope='conv3x3')
            x = layer_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        with tf.variable_scope('mis2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
                     scope='conv3x3')
            x = layer_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)
        # print(scope, 'x shape: ', x.get_shape().as_list())
        # print(scope, 'x_init shape: ', x_init.get_shape().as_list())
        return x + x_init

#平均池化+1×1卷积
def avg_conv(x, channels, use_bias=True, sn=True, scope='avg_conv'):
    with tf.variable_scope(scope):
        x = avg_pooling(x, kernel=2, stride=2)
        x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x


def conv_avg(x, channels, use_bias=True, sn=True, scope='conv_avg'):
    with tf.variable_scope(scope):
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = avg_pooling(x, kernel=2, stride=2)

        return x


def expand_concat(x, z):
    z = tf.reshape(z, shape=[z.shape[0], 1, 1, -1])
    z = tf.tile(z, multiples=[1, x.shape[1], x.shape[2], 1])  # expand
    x = tf.concat([x, z], axis=-1)

    return x


##################################################################################
# Sampling
##################################################################################

def down_sample(x):
    return avg_pooling(x, kernel=3, stride=2, pad=1)


def avg_pooling(x, kernel=2, stride=2, pad=0):
    if pad > 0:
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap


def z_sample(mean, logvar):
    eps = tf.random_normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mean + tf.exp(logvar * 0.5) * eps


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def batch_norm(x, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.999,
                                        center=True,
                                        scale=False,
                                        epsilon=0.001,
                                        scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake, content=False):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    if content:
        for i in range(n_scale):
            if type == 'lsgan':
                real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake[i]))

            if type == 'gan':
                real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

            loss.append(real_loss + fake_loss)

    else:
        for i in range(n_scale):
            if type == 'lsgan':
                real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake[i]))

            if type == 'gan':
                real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

            loss.append(real_loss * 1 + fake_loss)

    return sum(loss)


def generator_loss(type, fake, content=False):
    n_scale = len(fake)
    loss = []
    fake_loss = 0

    if content:
        for i in range(n_scale):
            if type == 'lsgan':
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 0.5))
            if type == 'gan':
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=0.5 * tf.ones_like(fake[i]), logits=fake[i]))
            loss.append(fake_loss)
    else:
        for i in range(n_scale):
            if type == 'lsgan':
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))
            if type == 'gan':
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

            loss.append(fake_loss)

    return sum(loss)


def l2_regularize(x):
    loss = tf.reduce_mean(tf.square(x))
    return loss


def kl_loss(mu, logvar):
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(y - x))
    return loss


def Fro_LOSS(batchimg):
    fro_norm = tf.square(tf.norm(batchimg, axis=[1, 2], ord='fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
    E = tf.reduce_mean(fro_norm)
    return E

def gradient(input):
    filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
    filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
    Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
    Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
    return Gradient

def grad_loss(img1, img2):
    # the input image is RGB image, and calculate the grad loss in R,G,B channel
    img1_1 = tf.expand_dims(img1[:, :, :, 0], -1)
    img1_2 = tf.expand_dims(img1[:, :, :, 1], -1)
    img1_3 = tf.expand_dims(img1[:, :, :, 2], -1)
    img2_1 = tf.expand_dims(img2[:, :, :, 0], -1)
    img2_2 = tf.expand_dims(img2[:, :, :, 1], -1)
    img2_3 = tf.expand_dims(img2[:, :, :, 2], -1)
    img1_1_grad = gradient(img1_1)
    img1_2_grad = gradient(img1_2)
    img1_3_grad = gradient(img1_3)
    img2_1_grad = gradient(img2_1)
    img2_2_grad = gradient(img2_2)
    img2_3_grad = gradient(img2_3)
    loss = (L1_loss(img1_1_grad, img2_1_grad) + L1_loss(img1_2_grad, img2_2_grad) + L1_loss(img1_3_grad, img2_3_grad)) / 3
    return loss
def Color_Fidelity_loss(img1, img2):
    R1 = tf.expand_dims(img1[:, :, :, 0], -1)
    G1 = tf.expand_dims(img1[:, :, :, 1], -1)
    B1 = tf.expand_dims(img1[:, :, :, 2], -1)
    R2 = tf.expand_dims(img2[:, :, :, 0], -1)
    G2 = tf.expand_dims(img2[:, :, :, 1], -1)
    B2 = tf.expand_dims(img2[:, :, :, 2], -1)
    CF_loss = L1_loss(tf.multiply(R1, G2), tf.multiply(R2, G1)) + \
              L1_loss(tf.multiply(R1, B2), tf.multiply(R2, B1)) + \
              L1_loss(tf.multiply(G1, B2), tf.multiply(G2, B1))
    return CF_loss

def gradient(input):
    filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
    filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
    Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
    Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
    return Gradient

def Gradient_loss(image_A, image_B):
    channel = image_A.get_shape().as_list()[-1]
    grad_loss = 0
    for i in range(0, channel):
        input_A = tf.expand_dims(image_A[:, :, i], -1)
        input_B = tf.expand_dims(image_B[:, :, i], -1)
        gradient_A = gradient(input_A)
        gradient_B = gradient(input_B)
        grad_loss = grad_loss + L1_loss(gradient_A, gradient_B)
    return grad_loss

def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                                trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
    return input_x_norm
def reorder_input(X, horizontal=False, reverse=False):
    # X.shape = (batch_size, row, column, channel)
    if horizontal:
        X = tf.transpose(X, perm=[2, 0, 1, 3])  #[column, batch_size, row, channel]
    else:
        X = tf.transpose(X, perm=[1, 0, 2, 3])  #[row, bath_size, column, channel]
    if reverse:
        X = tf.reverse(X, axis=[3])
    return X
def reorder_output(X, horizontal, reverse):
    if reverse:
        X = tf.reverse(X, axis=[3])
    if horizontal:
        # input: [column, batch_size, row, channel]
        X = tf.transpose(X, [1, 2, 0, 3]) # output: [bath_szie, row, column, channel]
    else:
        # input: [row, bath_size, column, channel]
        X = tf.transpose(X, [1, 0, 2, 3]) # output: [bath_szie, row, column, channel]
    return X


def compute(a, x):
    # x input is a tuple including input X and weight G
    H = a
    
    X, G = x
    L = H - X
    H = G * L + X
    return H

def LRNN_module(X, G, horizontal, reverse):
    X = reorder_input(X, horizontal, reverse)
    G = reorder_input(G, horizontal, reverse)
    initializer = tf.zeros_like(X[0])
    S = tf.scan(compute, (X, G), initializer)
    H = reorder_output(S, horizontal, reverse)
    return H