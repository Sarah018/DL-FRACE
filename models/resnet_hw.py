import torch.nn as nn
import math
import torch

__all__ = ['ResNet', 'resnet20', 'resnet56', 'resnet20_mnist', 'resnet20_mnist_s64', 'resnet20_letter', 'resnet20_letter_feature']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10):
        self.inplanes = inplanes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(inplanes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_MNIST(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10, sz_embedding=64, normalize_output=True):
        self.inplanes = inplanes
        super(ResNet_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(sz_embedding, num_classes)
        self.normalize_output = normalize_output
        self.sz_embedding = sz_embedding
        self.embedding_forward = self.embedding_layer(inplanes * 4, sz_embedding, weight_init=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_embedding_layer(self, in_features, sz_embedding, weight_init=None):
        embedding_layer = torch.nn.Linear(in_features, sz_embedding)
        if weight_init != None:
            weight_init(embedding_layer.weight)
        return embedding_layer

    def bn_inception_weight_init(self, weight):
        import scipy.stats as stats
        stddev = 0.001
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(
            X.rvs(weight.data.numel())
        ).resize_(weight.size())
        weight.data.copy_(values)

    def embedding_layer(self, in_features, sz_embedding=64, weight_init=True):
        embedding_layer = torch.nn.Linear(in_features, sz_embedding)
        if weight_init == True:
            self.bn_inception_weight_init(embedding_layer.weight)
        return embedding_layer


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feature = self.avgpool(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.embedding_forward(feature)
        if self.normalize_output == True:
            feature_norm = torch.nn.functional.normalize(feature, p=2, dim=1)
        x = self.fc(feature_norm)
        return x, feature


class ResNet_LETTER(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10):
        self.inplanes = inplanes
        super(ResNet_LETTER, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.fc = nn.Linear(inplanes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class ResNet_LETTER_FEATURE(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10, sz_embedding=64, normalize_output=True):
        self.inplanes = inplanes
        super(ResNet_LETTER_FEATURE, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.fc = nn.Linear(sz_embedding, num_classes)
        self.normalize_output = normalize_output
        self.sz_embedding = sz_embedding
        self.embedding_forward = self.embedding_layer(inplanes * 4, sz_embedding, weight_init=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_embedding_layer(self, in_features, sz_embedding, weight_init=None):
        embedding_layer = torch.nn.Linear(in_features, sz_embedding)
        if weight_init != None:
            weight_init(embedding_layer.weight)
        return embedding_layer

    def bn_inception_weight_init(self, weight):
        import scipy.stats as stats
        stddev = 0.001
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(
            X.rvs(weight.data.numel())
        ).resize_(weight.size())
        weight.data.copy_(values)

    def embedding_layer(self, in_features, sz_embedding=64, weight_init=True):
        embedding_layer = torch.nn.Linear(in_features, sz_embedding)
        if weight_init == True:
            self.bn_inception_weight_init(embedding_layer.weight)
        return embedding_layer


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feature = self.avgpool(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.embedding_forward(feature)
        if self.normalize_output == True:
            feature_norm = torch.nn.functional.normalize(feature, p=2, dim=1)
        x = self.fc(feature_norm)
        return x, feature


class ResNet_MNIST_S64(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10):
        self.inplanes = inplanes
        super(ResNet_MNIST_S64, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet20(inplanes, num_classes):
    model = ResNet(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model


def resnet56(inplanes, num_classes):
    model = ResNet(BasicBlock, [9, 9, 9], inplanes, num_classes)
    return model

def resnet20_mnist(inplanes, num_classes):
    model = ResNet_MNIST(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model

def resnet20_letter(inplanes, num_classes):
    model = ResNet_LETTER(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model

def resnet20_letter_feature(inplanes, num_classes):
    model = ResNet_LETTER_FEATURE(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model

def resnet20_mnist_s64(inplanes, num_classes):
    model = ResNet_MNIST_S64(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model