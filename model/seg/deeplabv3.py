import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP

class DeepLabFeatureHead(nn.Module):
    def __init__(self, in_channels: int) -> None:

        super().__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x
    
class DeepLabMultiHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.body_head = DeepLabFeatureHead(in_channels)
        self.joint_head = DeepLabFeatureHead(in_channels)

    def forward(self, x):
        body_feat = self.body_head(x)
        joint_feat = self.joint_head(x)
        return body_feat, joint_feat

class DeepLabV3(nn.Module):
    def __init__(self, type='resnet50', pretrained=True, num_joints=2):
        super(DeepLabV3, self).__init__()
        if type == 'resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(weights_backone=models.ResNet50_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        elif type == 'resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(weights_backone=models.ResNet101_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        elif type == 'mobilenet':
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights_backone=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        else:
            raise ValueError('Unknown type of DeepLabV3')
        
        self.model.classifier = DeepLabMultiHead(2048)
        
    def forward(self, x):
        # contract: features is a dict of tensors
        features = self.model.backbone(x)
        x = features["out"]
        f_body, f_joint = self.model.classifier(x)

        return f_body, f_joint
    
class DeepLabV3Flow(nn.Module):
    def __init__(self, type='resnet50', pretrained=True, num_joints=2):
        super(DeepLabV3Flow, self).__init__()
        if type == 'resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(weights_backone=models.ResNet50_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        elif type == 'resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(weights_backone=models.ResNet101_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        elif type == 'mobilenet':
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights_backone=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None, num_classes=num_joints)
        else:
            raise ValueError('Unknown type of DeepLabV3')
        
        self.model.classifier = DeepLabFeatureHead(2048)
        # self.model.backbone = None
        self.model.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        
        # contract: features is a dict of tensors
        features = self.model.backbone(x)

        # result = OrderedDict()
        x = features["out"]
        f_flow = self.model.classifier(x)

        return f_flow
    
if __name__ == '__main__':
    model = DeepLabV3(type='resnet50', pretrained=True, num_joints=17)
    x = torch.randn(8, 3, 256, 256)
    y, f = model(x)
    print(y.shape)
    print(f.shape)