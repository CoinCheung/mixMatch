#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from torch.nn import BatchNorm2d


'''
    As in the paper, the wide resnet only considers the resnet of the pre-activated version, and it only considers the basic blocks rather than the bottleneck blocks.
'''


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=0.001)
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = BatchNorm2d(out_chan, momentum=0.001)
        #  self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        #  residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(act1)

        out = shortcut + residual
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)



class WideResnetBackbone(nn.Module):
    def __init__(self, k=1, n=28):
        super(WideResnetBackbone, self).__init__()
        self.k = k
        self.n = n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6
        n_layers = [16,] + [self.k*16*(2**i) for i in range(3)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.ReLU(inplace=True)
        self.init_weight()

    def create_layer(self, in_chan, out_chan, bnum, stride=1):
        layers = [BasicBlockPreAct(in_chan, out_chan, stride=stride)]
        for _ in range(bnum-1):
            layers.append(BasicBlockPreAct(out_chan, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat) # 1/2
        feat4 = self.layer3(feat2) # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)
        return feat2, feat4

    def init_weight(self):
        for _, child in self.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    child.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not child.bias is None: nn.init.constant_(child.bias, 0)


class WideResnet(nn.Module):
    '''
    for wide-resnet-28-10, the definition should be WideResnet(n_classes, 10, 28)
    '''
    def __init__(self, n_classes, k=1, n=28):
        super(WideResnet, self).__init__()
        self.n_layers = n
        self.k = k
        self.backbone = WideResnetBackbone(k=k, n=n)
        self.classifier = nn.Linear(64*self.k, n_classes)
        self.bn = nn.BatchNorm1d(n_classes, momentum=0.001)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        feat = self.classifier(feat)
        feat = self.bn(feat)
        return feat

    def init_weight(self):
        nn.init.kaiming_normal_(
            self.classifier.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
        )
        if not self.classifier.bias is None:
            nn.init.constant_(self.classifier.bias, 0)

        #  for _, md in self.named_modules():
        #      if isinstance(md, BatchNorm2d):
        #          #  md.momentum = 1/(256**2)
        #          md.momentum = 0.1



if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    lb = torch.randint(0, 10, (2, )).long()

    net = WideResnetBackbone()
    out = net(x)
    print(out[0].size())
    del net, out

    net = WideResnet(n_classes=10)
    criteria = nn.CrossEntropyLoss()
    out = net(x)
    loss = criteria(out, lb)
    loss.backward()
    print(out.size())
