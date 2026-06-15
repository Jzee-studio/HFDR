import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, norm = False, mean = None, std = None):
        super(WideResNet, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class WideResNet_F(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, norm = False, mean = None, std = None):
        super(WideResNet_F, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.filter_size = 16
        self.Filter = SRMFilter(self.filter_size)
        self.Recon = Recalibration(self.filter_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_eval=False):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = self.conv1(x)
        HF, LF, mask = self.Filter(out)
        HF_fine = self.Recon(HF, mask)
        out = (HF_fine) + LF
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out_1 = self.fc(out)
        if is_eval == False:
            return out_1
        else:
            return out_1, mask

class WideResNet_LFCM(nn.Module):
    """
    WideResNet with SRMFilter + Recalibration + LFCM.
    LFCM canonicalizes the low-frequency component that would otherwise
    pass through unchanged in the standard HFDR architecture.
    """

    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0,
                 norm=False, mean=None, std=None,
                 codebook_size=64, code_dim=32, hidden_dim=64, tau=1.0):
        super(WideResNet_LFCM, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.filter_size = 16
        assert nChannels[0] == self.filter_size, \
            f"Conv1 output channels ({nChannels[0]}) must match filter_size ({self.filter_size})"
        self.Filter = SRMFilter(self.filter_size)
        self.Recon = Recalibration(self.filter_size)
        # LFCM canonicalizes the LF component
        self.LFCM = LFCM(in_channel=self.filter_size,
                         codebook_size=codebook_size,
                         code_dim=code_dim,
                         hidden_dim=hidden_dim,
                         tau=tau)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nw = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / nw))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_aux=False):
        """
        Args:
            x: input tensor (B, 3, 32, 32)
            return_aux: if True, returns (logits, aux_dict) where aux_dict contains
                        {'mask', 'z', 'z_hat', 'w'} for LFCM loss computation

        Returns:
            logits when return_aux=False (inference / PGD attack)
            (logits, aux_dict) when return_aux=True (training)
        """
        if self.norm:
            x = Normalization(x, self.mean, self.std)
        out = self.conv1(x)                          # (B, 16, 32, 32)
        HF, LF, mask = self.Filter(out)              # HF, LF: (B, 16, 32, 32)
        HF_fine = self.Recon(HF, mask)               # (B, 16, 32, 32)

        # LFCM canonicalizes LF features
        LF_canon, z, z_hat, w = self.LFCM(LF)        # LF_canon: (B, 16, 32, 32)

        out = HF_fine + LF_canon                     # (B, 16, 32, 32)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        logits = self.fc(out)

        if return_aux:
            aux = {'mask': mask, 'z': z, 'z_hat': z_hat, 'w': w}
            return logits, aux
        else:
            return logits


def WRN34_10(Num_class=10, Norm=False, norm_mean=None, norm_std=None):
    return WideResNet(num_classes=Num_class, depth=34, widen_factor=10, norm=Norm, mean=norm_mean, std=norm_std)

def WRN34_10_F(Num_class=10, Norm=False, norm_mean=None, norm_std=None):
    return WideResNet_F(num_classes=Num_class, depth=34, widen_factor=10, norm=Norm, mean=norm_mean, std=norm_std)

def WRN34_10_LFCM(Num_class=10, Norm=False, norm_mean=None, norm_std=None,
                  codebook_size=64, code_dim=32, hidden_dim=64, tau=1.0):
    return WideResNet_LFCM(num_classes=Num_class, depth=34, widen_factor=10,
                           norm=Norm, mean=norm_mean, std=norm_std,
                           codebook_size=codebook_size, code_dim=code_dim,
                           hidden_dim=hidden_dim, tau=tau)
