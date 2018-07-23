
import os
import numpy as np
import tensorflow as tf

# A simple function that adds two coordinate channels
def coord_conv(x):
    
    _, nw, nh, nc = tuple(x.shape.as_list())
    icrd = np.vstack([np.arange(nw) for _ in range(nh)])   / max(nw, nh)
    jcrd = np.vstack([np.arange(nh) for _ in range(nw)]).T / max(nw, nh)
    crd = np.stack([icrd, jcrd], axis=2)
    crd = crd.reshape([1, nw, nh, 2]).astype(dtype='float32')

    crd = tf.constant(crd)
    x_batch_size = tf.shape(x)[0]
    crd_tiled = tf.tile(crd, tf.stack([x_batch_size, 1, 1, 1]))
    x = tf.concat([x, crd_tiled], axis=3)
    return x

# A simple function to build a 3D convolutional layer
def build_conv3d(x, instructions):
    
    filters     = int(instruction[1])
    kernel_size = int(instruction[2])
    strides     = int(instruction[3])
    tfactivity  = tf.nn.relu

    x = tf.layers.conv3d(
            activation=tfactivity,
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='VALID')
    return x

# A simple function to build a 2D convolutional layer
# instructions will be off the form
def build_conv2d(x, instruction):
    
    if '+' in instruction[0]:
        x = coord_conv(x)
    filters     = int(instruction[1])
    kernel_size = int(instruction[2])
    strides     = int(instruction[3])
    tfactivity  = tf.nn.relu

    x = tf.layers.conv2d(
            activation=tfactivity,
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='VALID')
    return x

# Build a dense network
def build_dense(x, instruction):

    tfactivity=tf.nn.tanh
    dense = [ int(d) for d in instruction[1:] ]

    x = tf.layers.flatten(x)
    for d in dense:
        x = tf.layers.dense(x, d, activation=tfactivity)
    return x

# Build a dense network here as well, but this time, 
# provide special relu outputs for the final activation
# layer so that we can output values for the value function
# and the policy
def build_dense_vfp(x, n, instruction):

    tfactivity=tf.nn.tanh
    dense = [ int(d) for d in instruction[1:] ]

    x = tf.layers.flatten(x)
    for d in dense:
        x = tf.layers.dense(x, d, activation=tfactivity)
    x = tf.layers.dense(x, n, activation=tf.nn.relu)
    return x



class PolicyBuilder(object):
    """
A highly customizable Policy Builder Class
Help Dialogue

ex:
    c2d+:16:3:1_c2d:16:3:2_h:4096:2048_p:512_v:512

Lets you write a string syntax for specifying different models arch.
This builds a network in the following way.

Convolutional Network:

    c2d+:16:3:1
    It creates a Convolutional 2D Layer and uses coordConvs (which is
    marked by the '+' symbol).
    '16' represents the number of filters,
    '3'  represents the kernel size,
    '1'  represents the stride

    c2d:16:3:2
    It also creates a Convolutional 2D Layer but does not use coordConvs,
    (notice the lack of '+'). Again,
    '16' represents the number of filters,
    '3'  represents the kernel size,
    '2'  represents the stride

Fully Connected Layers:
    
    h:4096:2048
    This creates a bunch of hidden layers. Each number in the list
    represents the size of the hidden layers. These dense layers
    are shared by both the value and policy function

    pi:512
    This the dense layers that are unique to the policy function. The
    list of numbers represents the size of each hidden layer.

    vf:512
    This the dense layers that are unique to the value function. The
    list of numbers represents the size of each hidden layer.

The complete the string, you complete each of the individual parts
with underscore, like what is shown at the very top of this message.

WARNING: There is no error checkingcode here. You must makesure syntax
is correct. 

    """

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, build=None):
        
        self.sess = sess

        nw, nh, nc = ob_space
        nact = ac_space.n

        self.X = tf.placeholder(tf.float32, [None, nw, nh, nc]) #obs
        x = self.X

        # Construction of the model
        with tf.variable_scope("a2c_model", reuse=reuse):

            with tf.variable_scope('conv_layers'):
                for b in build['conv3d']:
                    x = build_conv3d(x, b)
            
                for b in build['conv2d']:
                    x = build_conv2d(x, b)
            
            with tf.variable_scope('hidden_layers'):
                h = build_dense(x, build['dense'][0])

            with tf.variable_scope('pi'):
                pi = build_dense_vfp(h, nact, build['pi'][0])

            with tf.variable_scope('v'):
                vf = build_dense_vfp(h, 1, build['vf'][0])[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(pi))
        self.a0 = tf.argmax(pi- tf.log(-tf.log(u)), axis=-1)
        self.nda = tf.argmax(pi, axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi,
            labels=one_hot_actions)

        self.pi = pi
        self.vf = vf

    def step(self, ob, stochastic=True):
        if stochastic: 
            return self.sess.run([self.a0, self.vf, self.neglogp0], {self.X:ob})
        else:
            return self.sess.run([self.nda, self.vf, self.neglogp0], {self.X:ob})

    def value(self, ob):
        v = self.sess.run(self.vf, {self.X:ob})
        return v

    def logits(self, ob):
        pi = self.sess.run([self.pi], {self.X: ob})
        return pi

    # Next two methods are required when we will have to generate the imaginations later in the I2A
    # code.
    def transform_input(self, X):
        return [X]

    def get_inputs(self):
        return [self.X]


def policy_parser(builder):
    builder = builder.split('_')
    builder = [ b.split(':') for b in builder ]
    bd = {
        'conv3d' : [],
        'conv2d' : [],
        'dense': [],
        'pi' : [],
        'vf' : [],
    }
    for b in builder:
        if 'c3d' in b[0]:
            bd['conv3d'].append(b)
        elif 'c2d' in b[0]:
            bd['conv2d'].append(b)
        elif 'h' in b[0]:
            bd['dense'].append(b)
        elif 'pi' in b[0]:
            bd['pi'].append(b)
        elif 'vf' in b[0]:
            bd['vf'].append(b)
        else:
            pass
    assert len(bd['dense']) == 1, "There should only be a single dense layer"
    assert len(bd['pi']) == len(bd['vf']) == 1, "Missing Value and Policy Function outputs"
    return bd

