import torch
import torchvision.models as models
import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ClassificationModel, self).__init__()
        if model_name == "ResNet":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
        if model_name == "Inception":
            self.backbone = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1
            )
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        print(self.backbone)

    def forward(self, x):
        x = self.backbone(x)
        return x


class ClassificationModelResNet(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModelResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
