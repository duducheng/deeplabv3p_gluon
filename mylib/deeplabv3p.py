from mxnet import gluon
from mxnet.gluon import nn


class DeepLabv3p(nn.HybridBlock):

    def __init__(self, OS=16, classes=21):
        super(DeepLabv3p, self).__init__()

        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        elif OS == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)
        else:
            raise ValueError

        self.entry_flow = EntryFlow(prefix="entry_flow_", entry_block3_stride=entry_block3_stride)

        middle_flow = nn.HybridSequential(prefix="middle_flow_")
        with middle_flow.name_scope():
            for i in range(16):
                middle_flow.add(XceptionBlock(filters_list=[728, 728, 728], kernel_size=3, strides=1,
                                              dilation=middle_block_rate, depth_activation=False, in_filters=728,
                                              prefix="unit_%s_" % (i + 1)))
        self.middle_flow = middle_flow

        exit_flow = nn.HybridSequential(prefix="exit_flow_")
        with exit_flow.name_scope():
            exit_flow.add(XceptionBlock(filters_list=[728, 1024, 1024], kernel_size=3, strides=1,
                                        use_shortcut_conv=True, dilation=exit_block_rates[0], depth_activation=False,
                                        in_filters=728, prefix="block1_"))
            exit_flow.add(XceptionBlock(filters_list=[1536, 1536, 2048], kernel_size=3, strides=1,
                                        dilation=exit_block_rates[1], depth_activation=True, in_filters=1024,
                                        use_shortcut=False, prefix="block2_"))
        self.exit_flow = exit_flow

        self.aspp = ASPP(atrous_rates=atrous_rates)

        skip_project = nn.HybridSequential()
        skip_project.add(nn.Conv2D(48, kernel_size=1, use_bias=False, prefix='feature_projection0_'))
        skip_project.add(nn.BatchNorm(prefix='feature_projection0_BN_', epsilon=1e-5))
        skip_project.add(nn.Activation("relu"))
        self.skip_project = skip_project

        decoder = nn.HybridSequential()
        decoder.add(SeparableConv(256, kernel_size=3, strides=1, dilation=1, depth_activation=True,
                                  in_filters=304, epsilon=1e-5, prefix='decoder_conv0_'))
        decoder.add(SeparableConv(256, kernel_size=3, strides=1, dilation=1, depth_activation=True,
                                  in_filters=256, epsilon=1e-5, prefix='decoder_conv1_'))
        decoder.add(nn.Conv2D(classes, kernel_size=1, use_bias=True, prefix='logits_semantic_'))
        self.decoder = decoder

    def hybrid_forward(self, F, x):
        *_, h, w = x.shape

        x, skip = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.aspp(x)

        x = F.contrib.BilinearResize2D(x, height=h // 4, width=w // 4)
        skip = self.skip_project(skip)
        x = F.concat(x, skip, dim=1)

        x = self.decoder(x)

        return F.contrib.BilinearResize2D(x, height=h, width=w)


class SeparableConv(nn.HybridBlock):

    def __init__(self, out_filters, kernel_size, strides, dilation, depth_activation, in_filters=None,
                 epsilon=1e-3, prefix=None):
        super(SeparableConv, self).__init__(prefix=prefix)

        if in_filters is None:
            in_filters = out_filters

        self.depth_activation = depth_activation

        padding = compute_same_padding(kernel_size, dilation)
        with self.name_scope():
            # filter_in==filter_out
            self.depthwise_conv = nn.Conv2D(in_filters, in_channels=in_filters, groups=in_filters,
                                            dilation=dilation, use_bias=False,
                                            padding=padding, strides=strides,
                                            kernel_size=kernel_size, prefix='depthwise_')
            self.bn1 = nn.BatchNorm(axis=1, epsilon=epsilon, prefix='depthwise_BN_')
            self.pointwise_conv = nn.Conv2D(out_filters, kernel_size=1, use_bias=False, prefix='pointwise_')
            self.bn2 = nn.BatchNorm(axis=1, epsilon=epsilon, prefix='pointwise_BN_')

    def hybrid_forward(self, F, x):
        if not self.depth_activation:
            x = F.relu(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        if self.depth_activation:
            x = F.relu(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        if self.depth_activation:
            x = F.relu(x)
        return x


class XceptionBlock(nn.HybridBlock):

    def __init__(self, filters_list, kernel_size, strides, dilation, depth_activation, in_filters=None,
                 use_shortcut_conv=None, prefix=None, return_skip=False, use_shortcut=True):
        super(XceptionBlock, self).__init__(prefix=prefix)

        assert len(filters_list) == 3

        if in_filters is None:
            in_filters = filters_list[0]

        if use_shortcut_conv is None:
            use_shortcut_conv = strides > 1

        with self.name_scope():
            self.conv1 = SeparableConv(filters_list[0], kernel_size, 1, dilation, depth_activation,
                                       in_filters=in_filters, prefix='separable_conv1_')
            self.conv2 = SeparableConv(filters_list[1], kernel_size, 1, dilation, depth_activation,
                                       in_filters=filters_list[0], prefix='separable_conv2_')
            self.conv3 = SeparableConv(filters_list[2], kernel_size, strides, dilation, depth_activation,
                                       in_filters=filters_list[1], prefix='separable_conv3_')
            if use_shortcut_conv:
                self.shortcut_conv = nn.Conv2D(filters_list[2], kernel_size=1, strides=strides, use_bias=False,
                                               prefix='shortcut_')
                self.shortcut_bn = nn.BatchNorm(axis=1, prefix='shortcut_BN_')

        self.use_shortcut_conv = use_shortcut_conv
        self.use_shortcut = use_shortcut
        self.return_skip = return_skip

    def hybrid_forward(self, F, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.conv3(x)
        if self.use_shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut)
        if self.return_skip:
            return x + shortcut, skip
        if self.use_shortcut:
            return x + shortcut
        return x


class EntryFlow(nn.HybridBlock):

    def __init__(self, prefix, entry_block3_stride):
        super(EntryFlow, self).__init__(prefix)

        with self.name_scope():
            self.conv1 = nn.HybridSequential(prefix='conv1_')
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(32, kernel_size=3, strides=2, padding=1, use_bias=False, prefix='1_'))
                self.conv1.add(nn.BatchNorm(axis=1, prefix='1_BN_'))
                self.conv1.add(nn.Activation("relu"))
            self.conv2 = nn.HybridSequential(prefix='conv1_')
            with self.conv2.name_scope():
                self.conv2.add(nn.Conv2D(64, kernel_size=3, padding=1, use_bias=False, prefix='2_'))
                self.conv2.add(nn.BatchNorm(axis=1, prefix='2_BN_'))
                self.conv2.add(nn.Activation("relu"))

            self.conv3 = XceptionBlock(filters_list=[128, 128, 128], kernel_size=3, strides=2,
                                       dilation=1, depth_activation=False, in_filters=64, prefix='block1_')
            self.conv4 = XceptionBlock(filters_list=[256, 256, 256], kernel_size=3, strides=2, return_skip=True,
                                       dilation=1, depth_activation=False, in_filters=128, prefix='block2_')
            self.conv5 = XceptionBlock(filters_list=[728, 728, 728], kernel_size=3, strides=entry_block3_stride,
                                       use_shortcut_conv=True, dilation=1, depth_activation=False, in_filters=256,
                                       prefix='block3_')

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, skip = self.conv4(x)
        x = self.conv5(x)
        return x, skip


class PoolRecover(nn.HybridBlock):

    def __init__(self):
        super(PoolRecover, self).__init__()

        self.gap = nn.HybridSequential()
        self.gap.add(nn.GlobalAvgPool2D())
        self.gap.add(nn.Conv2D(256, kernel_size=1, use_bias=False, prefix='image_pooling_'))
        self.gap.add(nn.BatchNorm(prefix='image_pooling_BN_', epsilon=1e-5))
        self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        *_, h, w = x.shape
        pool = self.gap(x)
        return F.contrib.BilinearResize2D(pool, height=h, width=w)


class ASPP(nn.HybridBlock):

    def __init__(self, atrous_rates):
        super(ASPP, self).__init__()

        b0 = nn.HybridSequential()
        b0.add(nn.Conv2D(256, kernel_size=1, use_bias=False, prefix='aspp0_'))
        b0.add(nn.BatchNorm(prefix='aspp0_BN_', epsilon=1e-5))
        b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = atrous_rates

        # rate = 6 (12)
        b1 = SeparableConv(256, kernel_size=3, strides=1, dilation=rate1, depth_activation=True,
                           in_filters=2048, epsilon=1e-5, prefix='aspp1_')
        # rate = 12 (24)
        b2 = SeparableConv(256, kernel_size=3, strides=1, dilation=rate2, depth_activation=True,
                           in_filters=2048, epsilon=1e-5, prefix='aspp2_')
        # rate = 18 (36)
        b3 = SeparableConv(256, kernel_size=3, strides=1, dilation=rate3, depth_activation=True,
                           in_filters=2048, epsilon=1e-5, prefix='aspp3_')

        b4 = PoolRecover()

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        self.concurent.add(b4)
        self.concurent.add(b0)
        self.concurent.add(b1)
        self.concurent.add(b2)
        self.concurent.add(b3)

        self.project = nn.HybridSequential()
        self.project.add(nn.Conv2D(256, kernel_size=1, use_bias=False, prefix='concat_projection_'))
        self.project.add(nn.BatchNorm(prefix='concat_projection_BN_', epsilon=1e-5))
        self.project.add(nn.Activation("relu"))
        # self.project.add(nn.Dropout(0.1))
        self.project.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))


def compute_same_padding(kernel_size, dilation):
    # TODO: compute `same` padding for stride<=2 ?
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end
