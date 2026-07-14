import torch.nn as nn
import torchvision.models as models


class ResNet34BinaryClassifier(nn.Module):
	def __init__(self, pretrained: bool = True, num_classes: int = 2):
		super().__init__()
		weights = models.ResNet34_Weights.DEFAULT if pretrained else None
		self.backbone = models.resnet34(weights=weights)
		in_features = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(in_features, num_classes)

	def forward(self, x):
		return self.backbone(x)


class VisionTransformerBinaryClassifier(nn.Module):
	def __init__(self, pretrained: bool = True, num_classes: int = 2, image_size: int = 224):
		super().__init__()
		weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
		self.backbone = models.vit_b_16(weights=weights, image_size=image_size)
		hidden_dim = self.backbone.heads.head.in_features
		self.backbone.heads.head = nn.Linear(hidden_dim, num_classes)

	def forward(self, x):
		return self.backbone(x)


def build_classifier(architecture: str = 'resnet34', pretrained: bool = True, num_classes: int = 2, image_size: int = 224):
	architecture = architecture.lower().replace('_', '-').strip()

	if architecture in {'resnet34', 'resnet-34'}:
		return ResNet34BinaryClassifier(pretrained=pretrained, num_classes=num_classes)

	if architecture in {'vit-b-16', 'vitb16', 'vit_b_16'}:
		return VisionTransformerBinaryClassifier(pretrained=pretrained, num_classes=num_classes, image_size=image_size)

	raise ValueError(f'Unsupported classification architecture: {architecture}')


def get_classifier_head(model: nn.Module):
	if hasattr(model.backbone, 'fc'):
		return model.backbone.fc
	if hasattr(model.backbone, 'heads') and hasattr(model.backbone.heads, 'head'):
		return model.backbone.heads.head
	raise AttributeError('Unsupported backbone: could not locate classifier head')


def freeze_backbone(model: nn.Module):
	for parameter in model.backbone.parameters():
		parameter.requires_grad = False
	for parameter in get_classifier_head(model).parameters():
		parameter.requires_grad = True


def unfreeze_backbone(model: nn.Module):
	for parameter in model.backbone.parameters():
		parameter.requires_grad = True
