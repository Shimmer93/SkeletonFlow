import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

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
        elif 'vgg' in type:
            self._init_vgg(type, pretrained)
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

        if type == 'resnet18':
            self.enc_dims = [64, 128, 256]
        else:
            self.enc_dims = [256, 512, 1024]

    def _init_vgg(self, type, pretrained):
        if type == 'vgg16_bn':
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        else:
            raise NotImplementedError
        vgg = nn.Sequential(*list(vgg.children()))
        self.enc1 = vgg[:23]
        self.enc2 = vgg[23:33]
        self.enc3 = vgg[33:43]

        self.enc_dims = [256, 512, 512]
    
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
            ConvBlock(self.enc_dims[2], self.enc_dims[2]),
            ConvBlock(self.enc_dims[2], self.enc_dims[2])
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
        # x = upsample(x, ratio=4)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_h, max_w):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        # create positional encoding matrix for height dimension
        pe_h = torch.zeros(max_h, d_model)
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        div_term_h = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_h)
        self.register_buffer('pe_h', pe_h)

        # create positional encoding matrix for width dimension
        pe_w = torch.zeros(max_w, d_model)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_w)
        self.register_buffer('pe_w', pe_w)

    def forward(self, x):
        # get shape of input tensor
        B, C, H, W = x.shape

        # create positional encoding for height dimension
        pe_h = self.pe_h[:H, :].T.unsqueeze(0).unsqueeze(-1)

        # create positional encoding for width dimension
        pe_w = self.pe_w[:W, :].T.unsqueeze(0).unsqueeze(-2)

        # concatenate positional encodings and add to input tensor
        x = x + pe_h.to(x.device) + pe_w.to(x.device)

        return x

class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels)
        )
        self.last_layer = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0, bn=False, relu=False)
        # self.pos_enc = PositionalEncoding(in_channels, 512, 512)
    
    def forward(self, x, return_features=False):
        # x = self.pos_enc(x)
        f = self.head(x)
        y = self.last_layer(f)

        if return_features:
            return y, f
        else:
            return y
        
class FeatureHead(nn.Module):
    def __init__(self, in_channels):
        super(FeatureHead, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels)
        )
    
    def forward(self, x):
        return self.head(x)

class UNet(nn.Module):
    def __init__(self, type='resnet50', pretrained=True, num_joints=15):
        super(UNet, self).__init__()
        self.encoder = Encoder(type=type, pretrained=pretrained)
        self.decoder = Decoder(self.encoder.enc_dims)
        self.head_body = FeatureHead(self.encoder.enc_dims[0])
        self.head_joint = FeatureHead(self.encoder.enc_dims[0])

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x = self.decoder(x1, x2, x3)
        f_body = self.head_body(x)
        f_joint = self.head_joint(x)
        return f_body, f_joint
    
class UNetFlow(nn.Module):
    def __init__(self, type='resnet50'):
        super(UNetFlow, self).__init__()
        self.encoder = Encoder(type=type, pretrained=False)
        self.decoder = Decoder(self.encoder.enc_dims)
        self.head_flow = FeatureHead(self.encoder.enc_dims[0])

        self.encoder.enc1[0] = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.std(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std
        x1, x2, x3 = self.encoder(x)
        x = self.decoder(x1, x2, x3)
        f = self.head_flow(x)
        return f