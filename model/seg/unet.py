import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def upsample(x, ratio=2):
    return F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True, bn=True, relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, type='resnet50', pretrained=True):
        super(Encoder, self).__init__()
        if 'resnet' in type:
            self._init_resnet(type, pretrained)
        else:
            raise NotImplementedError
    
    def _init_resnet(self, type, pretrained):
        if type == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif type == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif type == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        elif type == 'resnet152':
            resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        else:
            raise NotImplementedError
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.enc1 = resnet[:5]
        self.enc2 = resnet[5]
        self.enc3 = resnet[6]

        self.enc_dims = [256, 512, 1024]

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        return x1, x2, x3
    
class Decoder(nn.Module):
    def __init__(self, enc_dims):
        super(Decoder, self).__init__()
        self.enc_dims = enc_dims
        
        self.dec3 = nn.Sequential(
            ConvBlock(self.enc_dims[2], 2 * self.enc_dims[2]),
            ConvBlock(2 * self.enc_dims[2], self.enc_dims[2])
        )

        self.dec2 = nn.Sequential(
            ConvBlock(self.enc_dims[1] + self.enc_dims[2], self.enc_dims[2]),
            ConvBlock(self.enc_dims[2], self.enc_dims[1])
        )

        self.dec1 = nn.Sequential(
            ConvBlock(self.enc_dims[0] + self.enc_dims[1], self.enc_dims[1]),
            ConvBlock(self.enc_dims[1], self.enc_dims[0])
        )

    def forward(self, x1, x2, x3):
        x = self.dec3(x3)
        x = upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        x = upsample(x, ratio=4)
        return x
    
class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, out_channels, kernel_size=1, padding=0, bn=False, relu=False)
        )
    
    def forward(self, x):
        return self.head(x)

class UNet(nn.Module):
    def __init__(self, type='resnet50', pretrained=True, out_channels=2):
        super(UNet, self).__init__()
        self.encoder = Encoder(type=type, pretrained=pretrained)
        self.decoder = Decoder(self.encoder.enc_dims)
        self.head = Head(self.encoder.enc_dims[0], out_channels)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x = self.decoder(x1, x2, x3)
        x = self.head(x)
        return x