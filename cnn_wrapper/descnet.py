from cnn_wrapper.network import Network


class GeoDesc(Network):
    """GeoDesc definition."""

    def setup(self):
        (self.feed('data')
         .conv_bn(3, 32, 1, name='conv0')
         .conv_bn(3, 32, 1, name='conv1')
         .conv_bn(3, 64, 2, name='conv2')
         .conv_bn(3, 64, 1, name='conv3')
         .conv_bn(3, 128, 2, name='conv4')
         .conv_bn(3, 128, 1, name='conv5')
         .conv(8, 128, 1, biased=False, relu=False, padding='VALID', name='conv6')
         .l2norm(name='l2norm').squeeze(axis=[1, 2]))