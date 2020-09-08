import tensorflow as tf
import numpy as np

try:
    SummaryWriter = tf.train.SummaryWriter
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
except:
    SummaryWriter = tf.summary.FileWriter
    image_summary = tf.summary.image_summary
    scalar_summary = tf.summary.scalar_summary
    histogram_summary = tf.summary.histogram_summary
    merge_summary = tf.summary.merge_summary

def conv2D(x, output_dim, kernel_size=3, stride=1, bias=True, name="conv2D"):
    '''
    Convolution + bias
    Args:
        x: input tensor, tf.tensor
        output_dim: output dimension, int
        kernel_size: kernel size of conv (3x3 default), int
        stride: stride of convolution, int
        bias: whether add a bias or not, bool
        name: name of this tf.op, str
    '''
    input_shape = x.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size, kernel_size, input_shape[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2D(x, output_shape, kernel_size=3, stride=2, bias=True, name="deconv2D", with_w=False):
    '''
    Deconvolutional + bias
    Args:
        x: input tensor, tf.tensor
        output_shape, output tensor shape, list
        kernel_size: kernel size (3x3 default), int
        stride: stride to deconv, int
        bias: add bias or not, bool
        name: name of this tf.op
        with_w: whether return with weights and bias
    '''
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1]], 
            initializer=tf.random_normal_initializer(stddev=0.01))

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1])

        if bias:
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.1))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            if bias:
                return deconv, w, biases
            else:
                return deconv, w
        else:
            return deconv

def leaky_relu(x, leak=0.2, name="lrelu"):
    '''
    Leaky relu op
    '''
    return tf.maximum(x, leak*x)

def linear(x, output_size, scope=None, bias=True, with_w=False):
    '''
    FC layer
    Args:
        x: input tensor, tf.tensor
        output_size: output dimension, int
        scope: name scope of this layer
        bias: add bias or not, bool
        with_w: whether return with weights and bias
    '''
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.01))

        if bias:
            biases = tf.get_variable("biases", [output_size], initializer=tf.constant_initializer(0.1))
            y = tf.matmul(x, matrix) + biases
        else:
            y = tf.matmul(x, matrix)

        if with_w:
            if bias:
                return y, matrix, bias
            else:
                return y, matrix
        else:
            return y