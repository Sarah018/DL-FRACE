import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


# class Generator(nn.Module):
#     """Generator network."""
#
#     def __init__(self, conv_dim=64, c_dim=10, repeat_num=2):
#         super(Generator, self).__init__()
#
#         layers = []
#         layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=False))
#         layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
#         layers.append(nn.ReLU(inplace=True))
#
#         # Down-sampling layers.
#         curr_dim = conv_dim
#         for i in range(2):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2
#
#         # Bottleneck layers.
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
#
#         # Up-sampling layers.
#         for i in range(2):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim // 2
#
#         layers.append(nn.Conv2d(curr_dim, 1, kernel_size=5, stride=1, padding=2, bias=False))
#         layers.append(nn.Tanh())
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x, c):
#
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.repeat(1, 1, x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=10, repeat_num=2, emb_dim=784):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(1+1, conv_dim, kernel_size=5, stride=1, padding=2, bias=False)
        self.instanceNorm2d1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d2 = nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d3 = nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True)

        self.resBlock = ResidualBlock(dim_in=conv_dim*4, dim_out=conv_dim*4)

        self.deconv1 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.Conv2d(conv_dim, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.tanh = nn.Tanh()

        self.label_emb = nn.Embedding(c_dim, emb_dim)

        # layers = []
        # layers.append(nn.Conv2d(1, conv_dim, kernel_size=5, stride=1, padding=2, bias=False))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        # layers.append(nn.ReLU(inplace=True))
        #
        # # Down-sampling layers.
        # curr_dim = conv_dim
        # for i in range(2):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim * 2
        #
        # # Bottleneck layers.
        # for i in range(repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        #
        # # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2
        #
        # layers.append(nn.Conv2d(curr_dim, 1, kernel_size=5, stride=1, padding=2, bias=False))
        # layers.append(nn.Tanh())
        # self.main = nn.Sequential(*layers)
        # self.deconv1 = nn.ConvTranspose2d(c_dim, 32, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x, c):

        c = self.label_emb(c)
        c = c.view(c.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.deconv1(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.tanh(x)

        return x



class Generator_complexity(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=200, repeat_num=6):
        super(Generator_complexity, self).__init__()

        self.conv1 = nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.instanceNorm2d1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d2 = nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d3 = nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True)

        self.resBlock = ResidualBlock(dim_in=conv_dim*4, dim_out=conv_dim*4)

        self.deconv1 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

        self.label_emb = nn.Embedding(c_dim, 128*128*3)


    def forward(self, x, c):

        c = self.label_emb(c)
        c = c.view(c.size(0), 3, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.deconv1(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.tanh(x)

        return x



class Generator_complexity224(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=200, repeat_num=6):
        super(Generator_complexity224, self).__init__()

        self.conv1 = nn.Conv2d(3+3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.instanceNorm2d1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d2 = nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d3 = nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d4 = nn.InstanceNorm2d(conv_dim*8, affine=True, track_running_stats=True)

        self.resBlock = ResidualBlock(dim_in=conv_dim*8, dim_out=conv_dim*8)

        self.deconv1 = nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

        self.label_emb = nn.Embedding(c_dim, 224*224*3)


    def forward(self, x, c):

        c = self.label_emb(c)
        c = c.view(c.size(0), 3, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.instanceNorm2d4(x)
        x = self.relu(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.resBlock(x)
        x = self.deconv1(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.tanh(x)

        return x


class Generator_disentangle_EncoderCC224(nn.Module):
    """Generator network. disentanglement cross covariance"""

    def __init__(self, conv_dim=64, c_dim=200, repeat_num=6):
        super(Generator_disentangle_EncoderCC224, self).__init__()

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.instanceNorm2d1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d2 = nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d3 = nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.instanceNorm2d4 = nn.InstanceNorm2d(conv_dim*8, affine=True, track_running_stats=True)

        self.resBlock = ResidualBlock(dim_in=conv_dim*8, dim_out=conv_dim*8)

        self.conv5 = nn.Conv2d(conv_dim*8, c_dim, kernel_size=3, bias=False)
        self.avgpool = nn.AvgPool2d(26, stride=1)
        self.fc = nn.Linear(c_dim, c_dim)


    def forward(self, x):

        x = self.conv1(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.instanceNorm2d4(x)
        x = self.relu(x)
        x_content = self.resBlock(x)
        x_content = self.resBlock(x_content)
        x_content = self.resBlock(x_content)
        x_content = self.resBlock(x_content)
        x_content = self.resBlock(x_content)
        x_content = self.resBlock(x_content)

        x_class = self.conv5(x)
        x_class = self.avgpool(x_class)
        x_class = x_class.view(x_class.size(0), -1)
        x_class = self.fc(x_class)

        return x_content, x_class


class Generator_disentangle_DecoderCC224(nn.Module):
    """Generator network. disentanglement cross covariance"""

    def __init__(self, conv_dim=64, c_dim=200, repeat_num=6):
        super(Generator_disentangle_DecoderCC224, self).__init__()

        self.instanceNorm2d1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.instanceNorm2d2 = nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True)
        self.instanceNorm2d3 = nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True)

        self.deconv1 = nn.ConvTranspose2d(conv_dim * 8 + c_dim, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, x, c):
        # c standard label, not one-hot code
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.deconv1(x)
        x = self.instanceNorm2d3(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.instanceNorm2d2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.instanceNorm2d1(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.tanh(x)

        return x



class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, image_size=28, conv_dim=64, c_dim=10, repeat_num=2):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class Discriminator_complexity(nn.Module):
    """Discriminator network."""

    def __init__(self, image_size=224, conv_dim=64, c_dim=200, repeat_num=6):
        super(Discriminator_complexity, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))








def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        out_src = self.fc2(x)
        out_cls = self.fc(x)

        return out_src, out_cls


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            model.state_dict().update()

        # model.load_state_dict(state_dict)
    return model



def Discriminator_resnet18(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)