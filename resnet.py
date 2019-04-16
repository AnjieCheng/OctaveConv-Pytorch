import torch.nn as nn
from octave import *

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
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

    def __init__(self, block, layers, alpha, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []

        #stage 2
        self.layer2_1 = Residual_Unit_first(alpha=alpha, num_in=64, num_mid=64, num_out=256, first_block=True, stride=(1, 1))
        self.layer2_2 = Residual_Unit(alpha=alpha, num_in=256, num_mid=64, num_out=256, first_block=True, stride=(1, 1))
        self.layer2_3 = Residual_Unit(alpha=alpha, num_in=256, num_mid=64, num_out=256, first_block=True, stride=(1, 1))

        #stage 3
        self.layer3_1 = Residual_Unit(alpha=alpha, num_in=256, num_mid=128, num_out=512, first_block=True, stride=(2, 2))
        self.layer3_2 = Residual_Unit(alpha=alpha, num_in=512, num_mid=128, num_out=512, first_block=True, stride=(1, 1))
        self.layer3_3 = Residual_Unit(alpha=alpha, num_in=512, num_mid=128, num_out=512, first_block=True, stride=(1, 1))
        self.layer3_4 = Residual_Unit(alpha=alpha, num_in=512, num_mid=128, num_out=512, first_block=True, stride=(1, 1))

        #stage 4
        self.layer4_1 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(2, 2))
        self.layer4_2 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(1, 1))
        self.layer4_3 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(1, 1))
        self.layer4_4 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(1, 1))
        self.layer4_5 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(1, 1))
        self.layer4_6 = Residual_Unit(alpha=alpha, num_in=512, num_mid=256, num_out=1024, first_block=True, stride=(1, 1))

        #stage 5
        self.layer5_1 = Residual_Unit(alpha=alpha, num_in=1024, num_mid=512, num_out=2048, first_block=True, stride=(2, 2))
        self.layer5_2 = Residual_Unit(alpha=alpha, num_in=2048, num_mid=512, num_out=2048, first_block=True, stride=(1, 1))
        self.layer5_3 = Residual_Unit_last(alpha=alpha, num_in=2048, num_mid=512, num_out=2048, first_block=True, stride=(1, 1))


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        hf_x, lf_x = self.layer2_1(x)
        hf_x, lf_x = self.layer2_2(hf_x, lf_x)
        hf_x, lf_x = self.layer2_3(hf_x, lf_x)

        hf_x, lf_x = self.layer3_1(hf_x, lf_x)
        hf_x, lf_x = self.layer3_2(hf_x, lf_x)
        hf_x, lf_x = self.layer2_3(hf_x, lf_x)
        hf_x, lf_x = self.layer3_4(hf_x, lf_x)

        hf_x, lf_x = self.layer4_1(hf_x, lf_x)
        hf_x, lf_x = self.layer4_2(hf_x, lf_x)
        hf_x, lf_x = self.layer4_3(hf_x, lf_x)
        hf_x, lf_x = self.layer4_4(hf_x, lf_x)
        hf_x, lf_x = self.layer4_5(hf_x, lf_x)
        hf_x, lf_x = self.layer4_6(hf_x, lf_x)

        hf_x, lf_x = self.layer5_1(hf_x, lf_x)
        hf_x, lf_x = self.layer5_2(hf_x, lf_x)
        x = self.layer5_3(hf_x, lf_x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
