import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
import open_clip

class FrozenResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class FrozenViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()
        self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class FrozenCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.model.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)