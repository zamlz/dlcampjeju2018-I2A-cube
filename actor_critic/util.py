
import tensorflow as tf

# Helper functions for the Actor Critic algorithm

# Makes sure rewards past the 'done' aren't added to the reward expectation
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + (gamma * r * (1.0 - done))
        discounted.append(r)
    return discounted[::-1]


# This function is used when we loading from a different scope
def fix_tf_name(name, name_scope=None):
    if name_scope is not None:
        name = name[len(name_scope) + 1:]
    return name.split(':')[0]


def cat_entropy(logits):
    a = logits - tf.reduce_max(logits, 1, keepdims=True)
    e = tf.exp(a)
    z = tf.reduce_sum(e, 1, keepdims=True)
    p = e / z
    return tf.reduce_sum(p * (tf.log(z) - a), 1)
