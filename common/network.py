
import numpy as np
import tensorflow as tf

# A general class for building networks
class NetworkBuilder(object):
    """
A highly customizable Network Builder Class
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
   
    def __init__(self):
        pass

    def parser(self, builder):
        builder = builder.split('_')
        builder = [ b.split(':') for b in builder ]
        bd = {
            'conv3d' : [],
            'conv2d' : [],
            'dense': [],
            'pi' : [],
            'vf' : [],
            'en_z_de': [],
            'conv2dT': [],
            'conv3dT': [],
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
            elif 'z' in b[0]:
                bd['en_z_de'].append(b)
            elif 'c2dT' in b[0]:
                bd['conv2dT'].append(b)
            elif 'c3dT' in b[0]:
                bd['conv3dT'].append(b)
            else:
                pass
        return bd
       

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
def build_conv3d(x, instruction):
    
    conv_layer = tf.layers.conv3d
    if 'T' in instruction[0]:
        conv_layer = tf.layers.conv3d_transpose
    
    filters     = int(instruction[1])
    kernel_size = int(instruction[2])
    strides     = int(instruction[3])
    tfactivity  = tf.nn.relu

    x = conv_layer(
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

    conv_layer = tf.layers.conv2d
    if 'T' in instruction[0]:
        conv_layer = tf.layers.conv2d_transpose
        
    if '+' in instruction[0]:
        x = coord_conv(x)

    filters     = int(instruction[1])
    kernel_size = int(instruction[2])
    strides     = int(instruction[3])
    tfactivity  = tf.nn.relu

    x = conv_layer(
            activation=tfactivity,
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='VALID')
    return x

# Build a dense network
def build_dense(x, instruction):

    tfactivity=tf.nn.relu
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

    tfactivity=tf.nn.relu
    dense = [ int(d) for d in instruction[1:] ]

    x = tf.layers.flatten(x)
    for d in dense:
        x = tf.layers.dense(x, d, activation=tfactivity)
    x = tf.layers.dense(x, n, activation=None)
    return x

# Given some tensor x, concatenate the
# one-hot expanded of a with depth n to
# x and return it
# x : [None, d1, d2, ..., dn, c]
# a : [None]
# xa: [None, d1, d2, ..., dn, c+n]
def concat_actions(x, a, n):
 
    # Find other shapes to expand to
    a = tf.one_hot(a, depth=n, axis=-1)
    for s in x.shape.as_list()[1:-1][::-1]:
        a = tf.stack([a for _ in range(s)], axis=1)
    x = tf.concat([x,a], axis=-1)
    return x

# Sample from the distribution
def sample_z(self, mu, logvar):
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(logvar / 2) * eps
