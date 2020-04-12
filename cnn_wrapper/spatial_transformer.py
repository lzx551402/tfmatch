"""
This implementation is adpated from:
https://github.com/kevinzakka/spatial-transformer-network
"""

import tensorflow as tf

def _meshgrid(height, width):
    with tf.compat.v1.variable_scope('_meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

def transformer_crop(U, theta, out_size, stack_batch, name='SpatialTransformer', **kwargs):
    """
    Args:
        U: BxHxWxC
        theta: BxNx2x3
        out_size: (out_h, out_w)
        stack_batch: whether to stack along batch dim.
    Returns:
        output: (BxN)x(out_size)xC if stack_batch is true, else BxNx(out_size)xC
    """
    def _interpolate(im, x, y, num_kpt, out_size):
        with tf.compat.v1.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = im.get_shape()[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(height - 1, 'int32')
            max_x = tf.cast(width - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x_0 = tf.cast(tf.floor(x), 'int32')
            x_1 = x_0 + 1
            y_0 = tf.cast(tf.floor(y), 'int32')
            y_1 = y_0 + 1

            x_0 = tf.clip_by_value(x_0, zero, max_x)
            x_1 = tf.clip_by_value(x_1, zero, max_x)
            y_0 = tf.clip_by_value(y_0, zero, max_y)
            y_1 = tf.clip_by_value(y_1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = tf.tile(tf.expand_dims(tf.range(num_batch) * dim1, 1),
                           [1, num_kpt * out_height * out_width])
            base = tf.reshape(base, [-1])
            base_y0 = base + y_0 * dim2
            base_y1 = base + y_1 * dim2
            idx_a = base_y0 + x_0
            idx_b = base_y1 + x_0
            idx_c = base_y0 + x_1
            idx_d = base_y1 + x_1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x_0, 'float32')
            x1_f = tf.cast(x_1, 'float32')
            y0_f = tf.cast(y_0, 'float32')
            y1_f = tf.cast(y_1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _transform(theta, input_dim, out_size):
        with tf.compat.v1.variable_scope('_transform'):
            num_batch = tf.shape(theta)[0]
            num_kpt = tf.shape(theta)[1]
            num_channels = input_dim.get_shape()[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch * num_kpt]))
            grid = tf.reshape(grid, tf.stack([num_batch * num_kpt, 3, -1]))
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            t_g = tf.matmul(theta, grid)  # [BxN, 3, 3] * [BxN, 3, H*W]
            x_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, num_kpt, out_size)

            if stack_batch:
                output = tf.reshape(
                    input_transformed, tf.stack([num_batch * num_kpt, out_height, out_width, num_channels]))
            else:
                output = tf.reshape(
                    input_transformed, tf.stack([num_batch, num_kpt, out_height, out_width, num_channels]))
            return output
    with tf.compat.v1.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output