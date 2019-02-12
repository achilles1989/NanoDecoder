"""

"""
import torch.nn as nn
from onmt.encoders.encoder import EncoderBase
from torchvision.models.resnet import resnet18
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=(5,3), stride=(stride,1),
                     padding=(2,1), bias=True)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(5,3), stride=(stride,1),
                               padding=(2,1), bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    #     self.inplanes = 16
    #     super(ResNet, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,3), stride=(2,1), padding=(2,1),
    #                            bias=False)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.relu = nn.ReLU(inplace=True)
    #     # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 16, layers[0])
    #     # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
    #     # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
    #     # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
    #     self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    #     # self.conv_merge = nn.Conv2d(256*block.expansion, num_classes, kernel_size=(3,3), stride=1, padding=(0,1),bias=True)
    #     # self.avgpool = nn.AvgPool2d(5, stride=1)
    #     self.fc = nn.Linear(256 * block.expansion, num_classes)
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,3), stride=(2,1), padding=(2,1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                          kernel_size=1, stride=(stride,1), bias=False),
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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = x.squeeze(2).transpose(0, 1).transpose(0, 2).contiguous()
        x = self.fc(x)
        # x = self.conv_merge(x)
        # x = x.view(x.size(0), -1)
        # x = torch.squeeze()
        return x

class ResNetEncoder(EncoderBase):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, input_size, embeddings):
        super(ResNetEncoder, self).__init__()

        self.embeddings = embeddings
        self.hidden_size = hidden_size
        input_size = embeddings.embedding_size if self.embeddings else input_size
        # self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = ResNet(
            BasicBlock, cnn_kernel_width, num_classes=hidden_size
        )

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input) if self.embeddings else input
        s_len, batch, emb_dim = emb.size()
        emb_remap = emb.transpose(0, 1).transpose(1, 2).contiguous().view(batch, emb_dim,-1,s_len)

        # emb = emb.transpose(0, 1).contiguous()
        # emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        # emb_remap = shape_transform(emb_reshape)
        out = self.cnn(emb_remap).view(-1, batch, self.hidden_size)

        return emb_remap.transpose(0, 2).contiguous(), \
            out.transpose(0, 2).contiguous(), lengths


class ResNetForRNNEncoder(EncoderBase):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, dec_layers,
                 enc_rnn_size, dec_rnn_size,
                 cnn_kernel_width, input_size,
                 embeddings, rnn_type):
        super(ResNetForRNNEncoder, self).__init__()

        self.embeddings = embeddings
        self.dec_layers = dec_layers
        self.num_directions = 1
        input_size = embeddings.embedding_size if self.embeddings else input_size
        # self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = ResNet(
            BasicBlock, cnn_kernel_width, num_classes=enc_rnn_size
        )
        self.dec_rnn_size = dec_rnn_size
        self.dec_rnn_size_real = dec_rnn_size // self.num_directions
        self.rnn_type = rnn_type

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input) if self.embeddings else input
        s_len, batch, emb_dim = emb.size()
        emb_remap = emb.transpose(0, 1).transpose(1, 2).contiguous().view(batch, emb_dim,-1,s_len)

        out = self.cnn(emb_remap)

        # memory_bank = out.squeeze(2).transpose(0, 1).transpose(0, 2).contiguous()
        memory_bank = out.view(-1, batch, self.dec_rnn_size)

        state = memory_bank.new_full((self.dec_layers * self.num_directions,
                                      batch, self.dec_rnn_size_real), 0)
        if self.rnn_type == 'LSTM':
            # The encoder hidden is  (layers*directions) x batch x dim.
            encoder_final = (state, state)
        else:
            encoder_final = state
        return encoder_final, memory_bank, lengths