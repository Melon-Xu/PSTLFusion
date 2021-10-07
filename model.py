from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
# from losses import Perceptual_Loss
import h5py
import scipy.io as scio
import cv2
import numpy as np
import os


class HDFusionNet(object):
    def __init__(self, sess, args):
        self.model_name = 'HDFusionNet'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.num_attribute = args.num_attribute  # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.content_init_lr = args.lr * 5
        self.ch = args.ch
        self.concat = args.concat

        """ Weight """
        self.content_adv_w = args.content_adv_w
        self.domain_adv_w = args.domain_adv_w
        self.fake_w = args.fake_w
        self.recon_w = args.recon_w
        self.att_w = args.att_w
        self.kl_w = args.kl_w

        """ Generator """
        self.n_layer = args.n_layer
        self.n_z = args.n_z

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.n_d_con = args.n_d_con
        self.multi = True if args.n_scale > 1 else False
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        # self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainA_dataset = './dataset/{}'.format(self.dataset_name + '/trainA')
        # self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainB_dataset = './dataset/{}'.format(self.dataset_name + '/trainB')
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def fusion_strategy_select(self, fusion_strategy, ir_extracted_features, vi_extracted_features):
        if fusion_strategy == 'addition':
            fused_features = ir_extracted_features + vi_extracted_features
        elif fusion_strategy == 'mean':
            extended_ir_features = tf.expand_dims(ir_extracted_features, axis=-1)
            extended_vi_features = tf.expand_dims(vi_extracted_features, axis=-1)
            fused_features = tf.reduce_mean(tf.concat([extended_ir_features, extended_vi_features], axis=-1), axis=-1)
        elif fusion_strategy == 'max':
            extended_ir_features = tf.expand_dims(ir_extracted_features, axis=-1)
            extended_vi_features = tf.expand_dims(vi_extracted_features, axis=-1)
            fused_features = tf.reduce_max(tf.concat([extended_ir_features, extended_vi_features], axis=-1), axis=-1)
        return fused_features
        


    def content_encoder_ir(self, x, is_training=True, reuse=False, scope='content_encoder_ir'):
        feature_map = []
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_0_0')#输入与输出一致
            x = lrelu(x, 0.01)
            feature_map.append(x)#在下采样前将feature map保存至feature_map列表中，用于补充上采样过程中带来的信息丢失。
            #第一次下采样开始
            channel = channel * 2#self.ch*2
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_0_1')
            x = lrelu(x, 0.01)
            #第一次下采样结束
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_1_0')
            x = lrelu(x, 0.01)
            feature_map.append(x)
            #第二次下采样开始
            channel = channel * 2#self.ch * 4
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_1_1')
            x = lrelu(x, 0.01)
            #第二次下采样结束
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_2_0')
            x = lrelu(x, 0.01)
            x = conv(x, channel * 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_2_1')
            x = lrelu(x, 0.01)#self.ch * 8
        return x, feature_map

    def edge_encoder_ir(self, x, reuse=False, scope='edge_encoder_ir'):
        feature_map = []
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_0_0')#输出与输入尺寸一致
            x = lrelu(x, 0.01)
            feature_map.append(x)#在下采样前将feature map保存至feature_map列表中，用于补充上采样过程中带来的信息丢失。
            #第一次下采样开始
            channel = channel * 2#self.ch*2
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_0_1')
            x = lrelu(x, 0.01)
            #第一次下采样结束
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_1_0')
            x = lrelu(x, 0.01)
            feature_map.append(x)
            #第二次下采样开始
            channel = channel * 2#self.ch * 4
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_1_1')
            x = lrelu(x, 0.01)
            #第二次下采样结束
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_2_0')
            x = lrelu(x, 0.01)
            x = conv(x, channel * 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_2_1')
            x = lrelu(x, 0.01)#self.ch * 8
        #print('x shape: ', x.get_shape().as_list())
        return x, feature_map

    def edge_decoder_ir(self, x, f=None,reuse=False, scope="edge_decoder_ir"):
        change_features = []
        channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            #block1
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_0_0')
            x = lrelu(x, 0.01)
            change_features.append(x)
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_0_1')
            x = lrelu(x, 0.01)
            #block2
            #第一次上采样开始
            x = deconv(x, channel // 2, kernel=3, stride=2, scope= 'deconv_1_0')
            x = lrelu(x, 0.01)
            change_features.append(x)
            skip_feature = f[1]
            x = tf.concat([x, skip_feature], axis=-1)
            #第一次上采样结束
            x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_1_1')
            x = lrelu(x, 0.01)
            #block3
            #第二次上采样开始
            x = deconv(x, channel // 2, kernel=3, stride=2, scope= 'deconv_2_0')
            x = lrelu(x, 0.01)
            change_features.append(x)
            skip_feature = f[0]
            x = tf.concat([x, skip_feature], axis=-1)
            #第二次上采样结束
            x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_2_1')
            x = lrelu(x, 0.01)
        return x, change_features


    def content_decoder_ir(self, x, skip_features=None, change_features=None, reuse=False, scope="content_decoder_ir"):
        channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            #block1
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope = 'conv_0_0')
            x = lrelu(x, 0.01)
            x = x + change_features[0]
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope = 'conv_0_1')
            x = lrelu(x, 0.01)
            #block2
            x = deconv(x, channel // 2, kernel=3, stride=2, scope= 'deconv_1_0')
            x = lrelu(x, 0.01)
            x = x + change_features[1]
            x_channel = x.get_shape().as_list()[-1]
            skip_feature = skip_features[1]
            skip_feature = conv(skip_feature, x_channel, kernel=3, stride=1, pad=1, pad_type='reflect', 
                scope='identity_conv_0_0')
            x = x + skip_feature
            x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope = 'conv_1_1')
            x = lrelu(x, 0.01)
            #block3 
            x = deconv(x, channel // 2, kernel=3, stride=2, scope= 'deconv_2_0')
            x = lrelu(x, 0.01)
            x = x + change_features[2]
            x_channel = x.get_shape().as_list()[-1]
            skip_feature = skip_features[0]
            skip_feature = conv(skip_feature, x_channel, kernel=3, stride=1, pad=1, pad_type='reflect', 
                scope='identity_conv_1_0')
            x = x + skip_feature
            x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope = 'conv_2_1')
            x = lrelu(x, 0.01)

        return x

    def multiscale_module_vi(self, x, reuse=False, scope="multiscale_module_vi"):
        channel =16#每一次下采样之后的通道数
        input = x
        [_, H, W, _] = input.get_shape().as_list()
        with tf.variable_scope(scope, reuse=reuse):

            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect',
                scope='conv_0_0')
            x = lrelu(x, 0.01)

            block1_input = x
            #print("block1_input shape:", block1_input.get_shape().as_list())

            identity_conv_1 = conv(block1_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='identity_conv_1')
            #print("identity_conv_1 shape:", identity_conv_1.get_shape().as_list())
            #identity_conv_1 = lrelu(identity_conv_1,0.01)
            conv1 = conv(block1_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='conv_1_0')
            conv1 = lrelu(conv1, 0.01)
            conv2 = conv(conv1, channel, kernel=3, stride=1,pad=1,pad_type='reflect',
                scope='conv_1_1')
            conv2 = lrelu(conv2, 0.01)
            conv3 = conv(conv2, channel, kernel=1, stride=1,pad=0,pad_type='reflect',
                scope='conv_1_2')
            #print("conv3 shape: ", conv3.get_shape().as_list())
            block1_output = lrelu(identity_conv_1+conv3,0.01)

            block2_input = block1_output

            identity_conv_2 = conv(block2_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='identity_conv_2')
            #identity_conv_2 = lrelu(identity_conv_2,0.01)
            conv1 = conv(block2_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='conv_2_0')
            conv1 = lrelu(conv1, 0.01)
            conv2 = conv(conv1, channel, kernel=3, stride=1, pad=1, pad_type='reflect',
                scope='conv_2_1')
            conv2 = lrelu(conv2, 0.01)
            conv3 = conv(conv2, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='conv_2_2')
            block2_output = lrelu(identity_conv_2+conv3,0.01)

            block3_input = block2_output

            identity_conv_3 = conv(block3_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='identity_conv_3')
            #identity_conv_3 = lrelu(identity_conv_3,0.01)
            conv1 = conv(block3_input, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='conv_3_0')
            conv1 = lrelu(conv1, 0.01)
            conv2 = conv(conv1, channel, kernel=3, stride=1, pad=1, pad_type='reflect',
                scope='conv_3_1')
            conv2 = lrelu(conv2, 0.01)
            conv3 = conv(conv2, channel, kernel=1, stride=1, pad=0, pad_type='reflect',
                scope='conv_3_2')
            block3_output = lrelu(identity_conv_3+conv3, 0.01)
            multiscale_featrues = block3_output

            return multiscale_featrues

    def LRNN_ir(self, x, weights, reuse=False, scope='LRNN_ir'):
        #  LRNN_module(X, G, horizontal, reverse):
        with tf.variable_scope(scope, reuse=reuse):
            wx1 = weights[:, :, :, 0:16]
            wx2 = weights[:, :, :, 16:32]
            wy1 = weights[:, :, :, 32:48]
            wy2 = weights[:, :, :, 48:64]
            y1 = LRNN_module(x, wx1, horizontal=True, reverse=False)
            y2 = LRNN_module(x, wx1, horizontal=True, reverse=True)
            y3 = LRNN_module(x, wy1, horizontal=False, reverse=False)
            y4 = LRNN_module(x, wy1, horizontal=False, reverse=True)
            y5 = LRNN_module(y1, wx2, horizontal=True, reverse=False)
            y6 = LRNN_module(y2, wx2, horizontal=True, reverse=True)
            y7 = LRNN_module(y3, wy2, horizontal=False, reverse=False)
            y8 = LRNN_module(y4, wy2, horizontal=False, reverse=True)
            for i in range(y8.get_shape().as_list()[-1]):
                y_temp_5 = tf.expand_dims(y5[:, :, :, i], axis=-1)
                y_temp_6 = tf.expand_dims(y6[:, :, :, i], axis=-1)
                y_temp_7 = tf.expand_dims(y7[:, :, :, i], axis=-1)
                y_temp_8 = tf.expand_dims(y8[:, :, :, i], axis=-1)
                y_temp = tf.reduce_max(tf.concat([y_temp_5, y_temp_6, y_temp_7, y_temp_8], axis=-1), axis=-1)
                if i == 0:
                    y = tf.expand_dims(y_temp, axis=-1)
                else:
                    y = tf.concat([y, tf.expand_dims(y_temp, axis=-1)], axis=-1)
        print('y shape: ', y.get_shape().as_list())
        return y

    def fusion_module(self, content_stream, edge_stream, reuse=False, scope='fusion_module'):
        channel = 16
        with tf.variable_scope(scope, reuse=reuse):
            #stream = content_stream + edge_stream
            stream = tf.concat([content_stream, edge_stream], axis=-1)
            #stream = content_stream
            x = conv(stream, channel, kernel=3, stride=1, pad=1, pad_type='reflect',
                     scope='conv_' + str(0))
            x = lrelu(x, 0.01)
            x = conv(x, channels=self.img_ch, kernel=1, stride=1, scope='Fusion_logit')
            x = (tanh(x)+1) / 2
        return x


    ##################################################################################
    # Discriminator
    ##################################################################################

    def content_discriminator(self, x, reuse=False, scope='content_discriminator'):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch * self.n_layer
            for i in range(3):
                x = conv(x, channel, kernel=7, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                # x = layer_norm(x, scope='layer_norm_' + str(i))
                x = lrelu(x, 0.01)

            x = conv(x, channel, kernel=4, stride=1, scope='conv_3')
            x = lrelu(x, 0.01)
            x = conv(x, channels=1, kernel=1, stride=1, scope='D_content_logit')
            x = tf.clip_by_value(x, 1e-8, 2.0)
            D_logit.append(x)

            return D_logit

    def multi_discriminator(self, x_init, reuse=False, scope="multi_discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.01)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                             scope='ms_' + str(scale) + 'conv_' + str(i))
                    x = lrelu(x, 0.01)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='ms_' + str(scale) + 'D_logit')
                x = tf.clip_by_value(x, 1e-8, 2.0)#把张量中的数值限定在一个范围内，<1e-8时输出1e-8，>2.0时输出2.0，在中间时输出原值
                D_logit.append(x)

                x_init = down_sample(x_init)
            return D_logit

    def discriminator(self, x, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='D_logit')
            x = tf.clip_by_value(x, 1e-8, 2.0)
            print("discriminator x shape:", x.get_shape().as_list())
            D_logit.append(x)

            return D_logit

    def attribute_discriminator(self, x, reuse=False, scope="attribute_discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            x = attribute_connet(x, 64, use_bias=True, sn=True, scope='attribute_0')
            x = attribute_connet(x, 32, use_bias=True, sn=True, scope='attribute_1')
            x = attribute_connet(x, 1, use_bias=True, sn=True, scope='attribute_2')
            x = tf.clip_by_value(x, 1e-8, 1.0)
            D_logit.append(x)

        return D_logit

    ##################################################################################
    # Model
    ##################################################################################
    def discriminate_real(self, x_A, x_B):
        if self.multi:
            real_A_logit = self.multi_discriminator(x_A, scope='multi_discriminator_A')
            real_B_logit = self.multi_discriminator(x_B, scope='multi_discriminator_B')

        else:
            real_A_logit = self.discriminator(x_A, scope="discriminator_A")
            real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        if self.multi:
            fake_A_logit = self.multi_discriminator(x_ba, reuse=True, scope='multi_discriminator_A')
            fake_B_logit = self.multi_discriminator(x_ab, reuse=True, scope='multi_discriminator_B')

        else:
            fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
            fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def discriminate_content(self, content_A, content_B, reuse=False):
        content_A_logit = self.content_discriminator(content_A, reuse=reuse, scope='content_discriminator')
        content_B_logit = self.content_discriminator(content_B, reuse=True, scope='content_discriminator')
        return content_A_logit, content_B_logit

    def discriminate_attribute(self, attribute_A, attribute_B, reuse=False):
        attribute_B_logit = self.attribute_discriminator(attribute_B, reuse=reuse, scope='attribute_discriminator')
        attribute_A_logit = self.attribute_discriminator(attribute_A, reuse=True, scope='attribute_discriminator')
        return attribute_A_logit, attribute_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.content_lr = tf.placeholder(tf.float32, name='content_lr')

        """ Input Image"""

        self.domain_image = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_ch),
                                       name='domain_image')
        self.gradient_image = gradient(self.domain_image)
        #self.content_input = tf.concat([self.domain_ir, self.domain_vi], axis=-1)
        self.content_input_image = self.domain_image
        self.edge_input_image = tf.concat([self.domain_image, self.gradient_image], axis=-1)
        #self.edge_input = tf.concat([self.domain_vi, self.gradient_vi, self.domain_ir, self.gradient_ir], axis=-1)
        with tf.variable_scope('generator'):
            self.content_features_image, self.content_skip_featrues_image = self.content_encoder_ir(self.content_input_image)
            self.edge_features_image, self.edge_skip_featrues_image = self.edge_encoder_ir(self.edge_input_image)
            self.edge_weight_image, self.change_features_image = self.edge_decoder_ir(self.edge_features_image, self.edge_skip_featrues_image)
            self.content_stream_image = self.content_decoder_ir(self.content_features_image, self.content_skip_featrues_image, self.change_features_image)
            self.edge_multiscal_features_image = self.multiscale_module_vi(self.edge_input_image)
            self.content_stream = self.content_stream_image #self.fusion_strategy_select(fusion_strategy='max',ir_extracted_features=self.content_stream_ir,vi_extracted_features=self.content_stream_vi)
            self.edge_stream_image = self.LRNN_ir(self.edge_multiscal_features_image, self.edge_weight_image)
            # self.edge_stream_vi = self.LRNN_vi(self.edge_multiscal_features_vi, self.edge_weight_vi)
            self.edge_stream = self.edge_stream_image#self.fusion_strategy_select(fusion_strategy='addition',ir_extracted_features=self.edge_stream_ir,vi_extracted_features=self.edge_stream_vi)
            self.fusion_image = self.fusion_module(self.content_stream, self.edge_stream)


        ## Loss Function
        with tf.name_scope('g_loss'):
            
            self.p_loss = L1_loss(self.domain_image, self.fusion_image)
            self.grad_loss = L1_loss(gradient(self.domain_image), gradient(self.fusion_image))
            self.g_loss_total = 100 * (3 * self.p_loss +  7 * self.grad_loss)





            # tf.compat.v1.summary.scalar which is used to display scalar information
            # used to display loss
 
            # display total_loss
            tf.compat.v1.summary.scalar('p_loss', self.p_loss)
            # tf.compat.v1.summary.scalar('vi_p_loss', self.vi_p_loss)
            tf.compat.v1.summary.scalar('grad_loss', self.grad_loss)
            # tf.compat.v1.summary.scalar('vi_grad_loss', self.vi_grad_loss)
            tf.compat.v1.summary.scalar('loss_g', self.g_loss_total)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)
        with tf.name_scope('image'):
            tf.compat.v1.summary.image('input_image', tf.expand_dims(self.domain_image[1, :, :, :], 0))
            # tf.compat.v1.summary.image('ir_image', tf.expand_dims(self.domain_ir[1, :, :, :], 0))
            tf.compat.v1.summary.image('fusion_image', tf.expand_dims(self.fusion_image[1, :, :, :], 0))

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        variables_file = 'generator_variables.txt'
        if os.path.exists(variables_file):
            os.remove(variables_file)
        for var in G_vars:
            with open(variables_file, 'a') as log:
                log.write(var.name)
                log.write('\n')

        grads_G, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_total, G_vars), clip_norm=5)

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).apply_gradients(
            zip(grads_G, G_vars))

    def form_results(self, results_path='./Results'):
        """
        Forms folders for each run to store the tensorboard files, saved models and the log files.
        :return: three string pointing to tensorboard, saved models and log paths respectively.
        """
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        folder_name = "/{0}_{1}_model". \
            format('HDFusioNet', 'Pixel_Grad')
        tensorboard_path = results_path + folder_name + '/Tensorboard'
        log_path = results_path + folder_name + '/log'
        if not os.path.exists(results_path + folder_name):
            os.mkdir(results_path + folder_name)
            os.mkdir(tensorboard_path)
            os.mkdir(log_path)
        return tensorboard_path, log_path

    def train(self):
        # load train data
        # print('trainA_dataset :', self.trainA_dataset)
        dataset_name = 'train_TNO.h5'
        #dataset_name = 'train_RoadScene.h5'
        f = h5py.File(dataset_name, 'r')
        sources = f['data'][:]
        print(sources.shape)
        sources = np.transpose(sources, (0, 3, 2, 1))
        print('sources shape: ', sources.shape)
        num_imgs = sources.shape[0]
        # num_imgs = 800
        mod = num_imgs % self.batch_size
        n_batches = int(num_imgs // self.batch_size)
        print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
        self.iteration = n_batches


        if mod > 0:
            print('Train set has been trimmed %d samples...\n' % mod)
            sources = sources[:-mod]
        print("source shape:", sources.shape)
        # self.trainA_data = self.construct_train_data(self.trainA_dataset)
        # self.trainB_data = self.construct_train_data(self.trainB_dataset)

        batch_idxs = n_batches
        # initialize all variables
        tf.global_variables_initializer().run()
        self.summary_op = tf.summary.merge_all()
        tensorboard_path, log_path = self.form_results()
        log_name = os.path.join(log_path, 'log.txt')
        if os.path.exists(log_name):
            os.remove(log_name)
        self.writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)
        tf.initialize_all_variables().run()
        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        # self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / batch_idxs)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        content_lr = self.content_init_lr
        for epoch in range(start_epoch, self.epoch):
            np.random.shuffle(sources)
            if self.decay_flag:
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (
                        self.epoch - self.decay_epoch)  # linear decay
                content_lr = self.content_init_lr if epoch < self.decay_epoch else self.content_init_lr * (
                        self.epoch - epoch) / (self.epoch - self.decay_epoch)  # linear decay

            for idx in range(0, batch_idxs):
                patch_image = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0:1]
                train_feed_dict = {
                    self.lr: lr,
                    self.content_lr: content_lr,
                    self.domain_image: patch_image
                }

                _, summary_str, g_loss= self.sess.run(
                    [self.G_optim, self.summary_op,
                     self.g_loss_total], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                if idx % 50 == 0:
                    batch_images, batch_fusion_images, g_loss, p_loss, grad_loss = self.sess.run(
                        [self.domain_image, self.fusion_image,
                         self.g_loss_total, self.p_loss, self.grad_loss],
                        feed_dict=train_feed_dict)
                    print("Generator Loss:")
                    print("Epoch: [%2d/%2d] [%4d/%4d], total loss:[%.4f], pixel loss:[%.4f], grad loss:[%.4f]"
                          % (
                              epoch, self.epoch, idx, batch_idxs, g_loss, p_loss, grad_loss))
                    with open(log_name, 'a') as log:
                        log.write("Generator Loss:\n")
                        log.write("Epoch: [%2d/%2d] [%4d/%4d], total loss:[%.4f], pixel loss:[%.4f], grad loss:[%.4f]"
                          % (
                              epoch, self.epoch, idx, batch_idxs, g_loss, p_loss, grad_loss))
                        log.write('\n')
                if np.mod(idx + 1, self.print_freq) == 0:
                    save_images(batch_images, [self.batch_size, 1],
                                './{}/ir_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    #save_images(batch_vi_images, [self.batch_size, 1],
                                #'./{}/vi_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    save_images(batch_fusion_images, [self.batch_size, 1],
                                './{}/fusion_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))

                # display training status
                counter += 1

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.concat:
            concat = "_concat"
        else:
            concat = ""

        if self.sn:
            sn = "_sn"
        else:
            sn = ""

        return "{}{}_{}_{}layer_{}dis_{}scale{}".format(self.model_name, concat,
                                                                 self.gan_type,
                                                                 self.n_layer, self.n_dis, self.n_scale,
                                                                 sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # print("checkpoint dir:", checkpoint_dir)
        #通过checkpoint文件找到模型文件名
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # ckpt_name = r'checkpoint/DRIT_test_lsgan_4layer_4dis_3scale_5con_sn/DRIT.model-20281'
            # self.saver.restore(self.sess, ckpt_name)
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))
        checkpoint_dir = r'./checkpoint/DRIT_over2under_lsgan_4layer_4dis_3scale_5con_sn'
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # print(could_load, checkpoint_counter)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in test_A_files:  # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_attribute):
                image_path = os.path.join(self.result_dir, '{}_attribute{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_B, feed_dict={self.test_image: sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                            '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                            '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        for sample_file in test_B_files:  # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_attribute):
                image_path = os.path.join(self.result_dir, '{}_attribute{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                            '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                            '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")
        index.close()
        
        
    def guide_test(self):
        tf.global_variables_initializer().run()
        # test_dir = '/data/timer/comparsion/Dataset_resize//{}'.format(self.dataset_name)
        test_ir_dir = './Test_data/test_ir'
        test_vi_dir = './Test_data/test_vi'
        test_ir_dir_Road = './Test_data/Road_ir'
        test_vi_dir_Road = './Test_data/Road_vi'
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        """ Guided Image Translation """

        filelist = os.listdir(test_ir_dir_Road)#返回指定路径下的文件和文件夹列表。
        filelist.sort(key=lambda x: int(x[0:-4]))
        #filelist.sort(key=lambda x:int(x.split('.')[0]))
        result_dir = './Fusion_Results_RoadScene'
        self.fusion_dir = './Fusion_Results_RoadScene'        
        print(self.fusion_dir)
        check_folder(self.fusion_dir)
        
        #self.content_dir = './Edge_Results_TNO'
        self.detail_dir_ir = './Detail_Results_ir'
        self.detail_dir_vi = './Detail_Results_vi'
        check_folder(self.detail_dir_ir)
        check_folder(self.detail_dir_vi)
        for item in filelist:
            test_ir_file = os.path.join(os.path.abspath(test_ir_dir_Road), item)
            test_vi_file = os.path.join(os.path.abspath(test_vi_dir_Road), item)
            # sample_file_B = os.path.join(os.path.abspath(test_B_dir), item)
            num = item.split('.')[0]
            print(num)
            # self.sub_contet_dir_B = os.path.join(self.result_dir, self.model_dir, 'content_B', str(num))
            # check_folder(self.sub_contet_dir_B)
            # print('Processing A image: ' + sample_file_A)
            test_ir_image, h, w = load_test_data(test_ir_file, size=self.img_size)
            test_vi_image, h, w = load_test_data(test_vi_file, size=self.img_size)
            test_ir_image = np.asarray(test_ir_image)
            test_vi_image = np.asarray(test_vi_image)
            # sample_image_B, h_, w_ = load_test_data(sample_file_B, size=self.img_size)
            # sample_image_B = np.asarray(sample_image_B)
            #
            # fake_AA_path = os.path.join(self.fake_AA_dir, '{}'.format(os.path.basename(sample_file)))
            # fake_AB_path = os.path.join(self.fake_AB_dir, '{}'.format(os.path.basename(sample_file)))
            # fake_BB_path = os.path.join(self.fake_BB_dir, '{}'.format(os.path.basename(sample_file)))
            # fake_BA_path = os.path.join(self.fake_BA_dir, '{}'.format(os.path.basename(sample_file)))
            fusion_path = os.path.join(self.fusion_dir, '{}'.format(os.path.basename(test_ir_file) ))
            #print(fusion_path)
            self.sub_detail_dir_ir = os.path.join(self.detail_dir_ir, num)
            #print(self.sub_detail_dir_ir)
            check_folder(self.sub_detail_dir_ir)
            self.sub_detail_dir_vi = os.path.join(self.detail_dir_vi, num)
            check_folder(self.sub_detail_dir_vi)


            # guide_file = os.path.join(os.path.abspath(test_B_dir), str((int(num) % file_num) + 1) + '.png')
            # guide_file = os.path.join(os.path.abspath(test_B_dir), item)
            # guide_file = r'23/1.JPG'
            # guide_file = r'233.png'
            self.domain_ir = tf.placeholder(tf.float32, shape=(1, h, w, self.img_ch),
                                       name='domain_ir')
            self.domain_vi = tf.placeholder(tf.float32, shape=(1, h, w, self.img_ch),
                                        name='domain_vi')
            self.gradient_ir = gradient(self.domain_ir)
            self.gradient_vi = gradient(self.domain_vi)
            #self.content_input = tf.concat([self.domain_ir, self.domain_vi], axis=-1)
            #self.edge_input = tf.concat([self.domain_vi, self.gradient_vi, self.domain_ir, self.gradient_ir], axis=-1)
            self.content_input_ir = self.domain_ir
            self.content_input_vi = self.domain_vi
            self.edge_input_ir = tf.concat([self.domain_ir, self.gradient_ir], axis=-1)
            self.edge_input_vi = tf.concat([self.domain_vi, self.gradient_vi], axis=-1)
            start_time = time.time()
            with tf.variable_scope('generator'):
                self.content_features_ir, self.content_skip_featrues_ir = self.content_encoder_ir(self.content_input_ir, reuse=True)
                self.content_features_vi, self.content_skip_featrues_vi = self.content_encoder_ir(self.content_input_vi, reuse=True)
                self.edge_features_ir, self.edge_skip_featrues_ir = self.edge_encoder_ir(self.edge_input_ir, reuse=True)
                self.edge_features_vi, self.edge_skip_featrues_vi = self.edge_encoder_ir(self.edge_input_vi, reuse=True)
                self.edge_weight_ir, self.change_features_ir = self.edge_decoder_ir(self.edge_features_ir, self.edge_skip_featrues_ir, reuse=True)
                self.edge_weight_vi, self.change_features_vi = self.edge_decoder_ir(self.edge_features_vi, self.edge_skip_featrues_vi, reuse=True)
                self.content_stream_ir = self.content_decoder_ir(self.content_features_ir, self.content_skip_featrues_ir, self.change_features_ir, reuse=True)
                self.content_stream_vi = self.content_decoder_ir(self.content_features_vi, self.content_skip_featrues_vi, self.change_features_vi, reuse=True)
                self.edge_multiscal_features_ir = self.multiscale_module_vi(self.edge_input_ir, reuse=True)
                self.edge_multiscal_features_vi = self.multiscale_module_vi(self.edge_input_vi, reuse=True)
                self.content_stream = self.fusion_strategy_select(fusion_strategy='max',ir_extracted_features=self.content_stream_ir,vi_extracted_features=self.content_stream_vi)
                self.edge_stream_ir = self.LRNN_ir(self.edge_multiscal_features_ir, self.edge_weight_ir, reuse=True)
                self.edge_stream_vi = self.LRNN_ir(self.edge_multiscal_features_vi, self.edge_weight_vi, reuse=True)
                self.edge_stream = self.fusion_strategy_select(fusion_strategy='addition',ir_extracted_features=self.edge_stream_ir,vi_extracted_features=self.edge_stream_vi)
                self.fusion_image = self.fusion_module(self.content_stream, self.edge_stream, reuse=True)
            

            fusion_image, edge_ir, edge_vi = self.sess.run(
                    [self.fusion_image, self.edge_stream_ir, self.edge_stream_vi],
                    feed_dict={self.domain_ir: test_ir_image,
                               self.domain_vi: test_vi_image})
             
            
            #print('min : ', np.min(fusion_image))
            #print('max : ', np.max(fusion_image))
            
            fusion_image = (fusion_image - np.min(fusion_image)) / (np.max(fusion_image) - np.min(fusion_image))
            end_time = time.time()
            print("Testing Success! Testing time is [%f]"%(end_time-start_time))
            save_test_images(fusion_image, [1, 1], fusion_path)

            # edge_featrues = edge_featrues.squeeze()
            for convi_ir in range(np.size(edge_ir, -1)):
                    content_convi = edge_ir[0, :, :, convi_ir]
                    print(content_convi.shape)
                    content_convi = (content_convi - np.min(content_convi)) / (
                            np.max(content_convi) - np.min(content_convi))
                    content_convi = content_convi * 255
                    content_save_name = os.path.join(self.sub_detail_dir_ir, str(convi_ir + 1) + '.jpg')
                    print(content_save_name)
                    cv2.imwrite(content_save_name, content_convi)
            
            for convi_vi in range(np.size(edge_vi, -1)):
                    content_convi = edge_vi[0, :, :, convi_vi]
                    print(content_convi.shape)
                    content_convi = (content_convi - np.min(content_convi)) / (
                            np.max(content_convi) - np.min(content_convi))
                    content_convi = content_convi * 255
                    content_save_name = os.path.join(self.sub_detail_dir_vi, str(convi_vi + 1) + '.jpg')
                    print(content_save_name)
                    cv2.imwrite(content_save_name, content_convi)
