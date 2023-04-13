from ._utils import *

from torch import flatten, concat, nn


class ResUNetClassification(nn.Module):

    def __init__(self, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, export_FCL=False):
        super(ResUNetClassification, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.export_FCL = export_FCL

        self.expansion = 4
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
        self.conv = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoding1 = self._make_layer(64, layers[0])
        self.encoding2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.encoding3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.encoding4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.decoding3_up, self.decoding3 = self._make_layer_up(256, layers[4], concatnate=False)
        self.decoding2_up, self.decoding2 = self._make_layer_up(128, layers[5], concatnate=True)
        self.decoding1_up, self.decoding1 = self._make_layer_up(64, layers[6], concatnate=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(64 * self.expansion * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_up(self, planes, blocks, concatnate=False, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation

        if concatnate:
            inplanes = self.inplanes*2
        else:
            inplanes = self.inplanes
        
        layer_up = BottleneckUpsample(inplanes, int(self.inplanes/self.expansion/2), stride, self.groups,
                        self.base_width, previous_dilation, norm_layer)
        self.inplanes = planes * self.expansion
        
        layers = []
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes*2, planes*2, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layer_up, nn.Sequential(*layers)

    def _forward_impl(self, input_tensor):
        
        conv = self.conv(input_tensor)
        bn = self.bn(conv)
        relu = self.relu(bn)
        maxpool = self.maxpool(relu)

        encoding1 = self.encoding1(maxpool)
        encoding2 = self.encoding2(encoding1)
        encoding3 = self.encoding3(encoding2)
        encoding4 = self.encoding4(encoding3)
        decoding3_up = self.decoding3_up(encoding3.size(), encoding4)
        decoding3 = self.decoding3(concat([encoding3, decoding3_up], axis=1))
        decoding2_up = self.decoding2_up(encoding2.size(), decoding3)
        decoding2 = self.decoding2(concat([encoding2, decoding2_up], axis=1))
        decoding1_up = self.decoding1_up(encoding1.size(), decoding2)
        decoding1 = self.decoding1(concat([encoding1, decoding1_up], axis=1))

        avgpool = self.avgpool(decoding1)
        _flatten = flatten(avgpool, 1)
        fully_connected = self.fully_connected(_flatten)

        if self.export_FCL:
            return decoding1, fully_connected
        else:
            return fully_connected

    def forward(self, x):
        return self._forward_impl(x)

class ResUNetSegmentation(nn.Module):

    def __init__(self, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResUNetSegmentation, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.expansion = 4
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
        self.conv = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoding1 = self._make_layer(64, layers[0])
        self.encoding2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.encoding3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.encoding4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.decoding3_up, self.decoding3 = self._make_layer_up(256, layers[4], concatnate=False)
        self.decoding2_up, self.decoding2 = self._make_layer_up(128, layers[5], concatnate=True)
        self.decoding1_up, self.decoding1 = self._make_layer_up(64, layers[6], concatnate=True)
        self.final_layer = self._make_final_layer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_up(self, planes, blocks, concatnate=False, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation

        if concatnate:
            inplanes = self.inplanes*2
        else:
            inplanes = self.inplanes
        
        layer_up = BottleneckUpsample(inplanes, int(self.inplanes/self.expansion/2), stride, self.groups,
                        self.base_width, previous_dilation, norm_layer)
        self.inplanes = planes * self.expansion
        
        layers = []
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes*2, planes*2, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layer_up, nn.Sequential(*layers)
    
    def _make_final_layer(self):
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        layers = []
        layers.append(self.final_conv)
        layers.append(self.sigmoid)

        return nn.Sequential(*layers)

    def _forward_impl(self, input_tensor):
        
        conv = self.conv(input_tensor)
        bn = self.bn(conv)
        relu = self.relu(bn)
        maxpool = self.maxpool(relu)

        encoding1 = self.encoding1(maxpool)
        encoding2 = self.encoding2(encoding1)
        encoding3 = self.encoding3(encoding2)
        encoding4 = self.encoding4(encoding3)
        decoding3_up = self.decoding3_up(encoding3.size(), encoding4)
        decoding3 = self.decoding3(concat([encoding3, decoding3_up], axis=1))
        decoding2_up = self.decoding2_up(encoding2.size(), decoding3)
        decoding2 = self.decoding2(concat([encoding2, decoding2_up], axis=1))
        decoding1_up = self.decoding1_up(encoding1.size(), decoding2)
        decoding1 = self.decoding1(concat([encoding1, decoding1_up], axis=1))
        final_layer = self.final_layer(decoding1)

        return final_layer

    def forward(self, x):
        return self._forward_impl(x)