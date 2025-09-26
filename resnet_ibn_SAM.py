import functools
import math
import warnings
import torch
import softpool_cuda
from pytorch.SoftPool import soft_pool2d,SoftPool2d
from torchvision.models import ResNet

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from modules import IBN
import torch.nn as nn
__all__ = ['ResNet_IBN',  'resnet50_ibn_b']



model_urls = {
    'resnet18_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'resnet34_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,attention_module=None):
        super(Bottleneck_IBN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups


        if ibn == 'a':
            self.bn1 = IBN(width)
        else:
            self.bn1 = norm_layer(width)

        self.conv1 = conv1x1(inplanes,width)

        self.bn2 = norm_layer(width)
        self.conv2 =conv3x3(width, width, stride,groups,dilation)


        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.conv3 = conv1x1(width, planes * self.expansion)

        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        # self.relu1 = nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                self.conv2,
                self.simam
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual


        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attention_module=None,
                 deep_stem=False, stem_width=32, avg_down=False,use_softpool=True): #2022.6.30日加入zero-init

        super(ResNet_IBN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = stem_width*2 if deep_stem else 64
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(3, stem_width, stride=2),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(self.inplanes if not deep_stem else stem_width*2, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(self.inplanes if not deep_stem else stem_width*2)
        self.relu = nn.ReLU(inplace=True)

        if not use_softpool:
            self.pool= nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        else:
            self.pool = SoftPool2d(kernel_size=(2,2),stride=(2,2),force_inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0], avg_down=avg_down,
                                       attention_module=attention_module)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], ibn=ibn_cfg[1], avg_down=avg_down,
                                       attention_module=attention_module)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], ibn=ibn_cfg[2], avg_down=avg_down,
                                       attention_module=attention_module)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], ibn=ibn_cfg[3], avg_down=avg_down,
                                       attention_module=attention_module)


        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_IBN):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,ibn=None,avg_down=False,attention_module=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, count_include_pad=False, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    norm_layer(planes * block.expansion)
                )

        layers = []

        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample,self.groups,
                            self.base_width, previous_dilation, norm_layer,attention_module))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and _ < blocks-1) else ibn,
                                groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,attention_module=attention_module
                                ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)


        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.reshape(x.size(0),-1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        new_state_dict = model.state_dict()
        new_state_dict.update(state_dict)
        model.load_state_dict(new_state_dict)
    return model


def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       use_softpool=False,deep_stem=True, stem_width=32, avg_down=True,
                       attention_module=simam_module,
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b']),False)
    return model

