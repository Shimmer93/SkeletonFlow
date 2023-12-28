import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP

class DeepLabHeadWithFeature(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:

        super().__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # feat = x
        # x = self.conv2(x)
        # return x, feat
        return x
    
class DeepLabMultiHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.body_head = DeepLabHeadWithFeature(in_channels, 1)
        self.joint_head = DeepLabHead(in_channels, num_classes)

    def forward(self, x):
        body_feat = self.body_head(x)
        joint = self.joint_head(x)
        return joint, body_feat

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
        
        self.model.classifier = DeepLabMultiHead(2048, num_joints)
        
    def forward(self, x):
        # def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.model.backbone(x)

        # result = OrderedDict()
        x = features["out"]
        x, f = self.model.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        # result["out"] = x

        # if self.aux_classifier is not None:
        #     x = features["aux"]
        #     x = self.aux_classifier(x)
        #     x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            # result["aux"] = x

        return x, f
    
if __name__ == '__main__':
    model = DeepLabV3(type='resnet50', pretrained=True, num_joints=17)
    x = torch.randn(8, 3, 256, 256)
    y, f = model(x)
    print(y.shape)
    print(f.shape)