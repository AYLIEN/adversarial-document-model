import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def vectors(model, data, session):
    vecs = []
    for _, x in data:
        vecs.extend(
            session.run([model.rep], feed_dict={
                model.x: x,
                model.mask: np.ones_like(x)
            })[0]
        )
    return np.array(vecs)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def linear(input, output_dim, scope=None, stddev=None):
    if stddev:
        norm = tf.random_normal_initializer(stddev=stddev)
    else:
        norm = tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / input.get_shape()[1].value)
        )
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=norm
        )
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def generator(z, size, output_size):
    h0 = tf.nn.relu(slim.batch_norm(linear(z, size, 'h0')))
    h1 = tf.nn.relu(slim.batch_norm(linear(h0, size, 'h1')))
    return tf.nn.sigmoid(linear(h1, output_size, 'h2'))


def discriminator(x, mask, size):
    noisy_input = x * mask
    h0 = leaky_relu(linear(noisy_input, size, 'h0'))
    h1 = linear(h0, x.get_shape()[1], 'h1')
    diff = x - h1
    return tf.reduce_mean(tf.reduce_sum(diff * diff, 1)), h0


class ADM(object):
    def __init__(self, x, z, mask, params):
        self.x = x
        self.z = z
        self.mask = mask

        with tf.variable_scope('generator'):
            self.generator = generator(z, params.g_dim, params.vocab_size)

        with tf.variable_scope('discriminator'):
            self.d_loss, self.rep = discriminator(x, mask, params.z_dim)

        with tf.variable_scope('discriminator', reuse=True):
            self.g_loss, _ = discriminator(self.generator, mask, params.z_dim)

        margin = params.vocab_size // 20
        self.d_loss += tf.maximum(0.0, margin - self.g_loss)

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('discriminator')]
        self.g_params = [v for v in vars if v.name.startswith('generator')]

        step = tf.Variable(0, trainable=False)

        self.d_opt = gradients(
            opt=tf.train.AdamOptimizer(
                learning_rate=params.learning_rate,
                beta1=0.5
            ),
            loss=self.d_loss,
            vars=self.d_params,
            step=step
        )

        self.g_opt = gradients(
            opt=tf.train.AdamOptimizer(
                learning_rate=params.learning_rate,
                beta1=0.5
            ),
            loss=self.g_loss,
            vars=self.g_params,
            step=step
        )
