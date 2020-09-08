from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *

class BuddhaGAN(object):
    def __init__(self, sess, 
            batch,
            input_height=240,
            input_width=240,
            output_height=240,
            output_width=240,
            z_dim=10,
            gf_dim=64,
            df_dim=64,
            c_dim=3,
            max_to_keep=3,):
        '''
        Args:
            sess: tf.session
            batch: batch size
            input_height: 输入Generator的height
            input_width: 输入Generator的width
            output_height: 输出图片的height
            output_width: 输出图片的width
            z_dim: 采样噪声的维度
            gf_dim: 生成器第一层的filter深度
            df_dim：识别器第一层的filter深度
            c_dim: 图片dimension，3 for RGB
        '''
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.sess = sess
        self.batch = batch
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.max_to_keep = max_to_keep

        self.input_ = tf.placeholder(tf.float32, [self.batch, self.input_height, self.input_width, 3], "input_") # input images
        self.z = tf.placeholder(tf.float32, [self.batch, self.z_dim], "z") # noise, sample from Pz
        self.y = tf.placeholder(tf.float32, [self.batch, None], "y") # label
        self.build_model()

    def build_model(self):
        self.G = self.generator(self.z) # generated image
        self.D_real = self.discriminator(self.input_, self.y, reuse=False)
        self.D_fake = self.discriminator(self.G, self.y, reuse=True)
        losses = self.loss_func(self.G, self.D_real, self.D_fake)
        self.D_real_loss = losses[0]
        self.D_fake_loss = losses[1]
        self.D_loss = losses[0] + losses[1]
        self.G_loss = losses[2]
        self.total_loss = self.D_losss + self.G_loss

        self.z_sum = histogram_summary("z", self.z)
        self.d_real_sum = histogram_summary("D_real", self.D_real)
        self.d_fake_sum = histogram_summary("D_fake", self.D_fake)
        self.g_sum = image_summary("G", self.G)
        self.d_real_loss_sum = scalar_summary("D_real_loss", self.D_real_loss)
        self.d_fake_loss_sum = scalar_summary("D_fake_loss", self.D_fake_loss)
        self.d_loss_sum = scalar_summary("D_loss", self.D_loss)
        self.g_loss_sum = scalar_summary("G_loss", self.G_loss)

        t_vars = tf.trainable_variables()
        self.D_vars = [var for var in t_vars if 'd_' in var.name]
        self.G_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def discriminator(self, x, y, reuse=False):
        """
        x: image, [batch, w, h, c_dim]
        y: label, [batch, 1]
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = leaky_relu(conv2D(x, self.df_dim, name='d_h0_conv'))
            h1 = leaky_relu(conv2D(h0, self.df_dim*2, name='d_h1_conv'))
            h2 = leaky_relu(conv2D(h1, self.df_dim*4, name='d_h2_conv'))
            h3 = leaky_relu(conv2D(h2, self.df_dim*8, name='d_h3_conv'))
            h4 = linear(tf.reshape(h3, [self.batch, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4)

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_w, s_h = self.output_width, self.output_height
            s_w2, s_h2 = conv_out_size_same(s_w, 2), conv_out_size_same(s_h, 2)
            s_w4, s_h4 = conv_out_size_same(s_w2, 2), conv_out_size_same(s_h2, 2)
            s_w8, s_h8 = conv_out_size_same(s_w4, 2), conv_out_size_same(s_h4, 2)
            s_w16, s_h16 = conv_out_size_same(s_w8, 2), conv_out_size_same(s_h8, 2)

            # project z and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, s_w16*s_h16*self.gf_dim*8, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s_w16, s_h16, self.gf_dim*8])
            h0 = leaky_relu(self.h0)

            h1, h1_w, h1_b = deconv2D(h0, [self.batch, s_w8, s_h8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = leaky_relu(h1)

            h2, h2_w, h2_b = deconv2D(h1, [self.batch, s_w4, s_h4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = leaky_relu(h2)

            h3, h3_w, h3_b = deconv2D(h2, [self.batch, s_w2, s_h2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = leaky_relu(h3)

            h4, h4_w, h4_b = deconv2D(h3, [self.batch, s_w, s_h, self.c_dim], name='g_h4', with_w=True)
            
            return tf.nn.tanh(h4)

    def loss_func(self, G, D_real, D_fake):
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        d_loss_real = sigmoid_cross_entropy_with_logits(D_real, tf.ones_like(D_real))
        d_loss_fake = sigmoid_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake))
        # d_loss = d_loss_real + d_loss_fake
        g_loss = sigmoid_cross_entropy_with_logits(G, tf.zeros_like(G))
        return [d_loss_real, d_loss_fake, g_loss]

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.D_loss, var_list=self.D_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.G_loss, var_list=self.G_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initializer_all_variables().run()

        # summary in log file
        self.G_sum = merge_summary([self.z_sum, self.d_fake_sum, self.g_sum, self.d_fake_loss_sum, self.g_loss_sum])
        self.D_sum = merge_summary([self.z_sum, self.d_real_sum, self.d_real_loss_sum, self.d_loss_sum])
        self.write = SummaryWriter(os.path.join(config.out_dir, 'logs'), self.sess.graph)

        # generate noise
        sample_z = gen_random(size=[self.batch, self.z_dim])

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" === loading checkpoint Success ===")
        else:
            print(" === loading checkpoint Fail === ")

        for epoch in xrange(config.epoch):
            pass
        # update D network
        _, summary_str = self.sess.run([d_optim, ])

    def inference(self, config):
        raise NotImplementedError

    def load(self, ckpt_dir):
        print(" === loading checkpoint: {} ===".format(ckpt_dir))
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(ckpt_dir, ckpt_name))
            counter = ckpt_name.split('-')[-1]
            try:
                counter = int(counter)
            except:
                counter = 1
            print(" === Success load checkpoint: {} ===".format(ckpt_name))
            return True, counter
        else:
            print(" === Failed load checkpoint ===")
            return False, 0
