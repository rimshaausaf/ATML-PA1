import torch
import torch.nn as nn
class PROSERResNet(nn.Module):
    def __init__(self, base_model, num_known=10, num_dummy=5):
        super().__init__()
        self.conv1, self.bn1, self.relu = base_model.conv1, base_model.bn1, base_model.relu
        self.layer1, self.layer2, self.layer3, self.layer4 = base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_known + num_dummy)
        with torch.no_grad():
            self.fc.weight[:num_known].copy_(base_model.fc.weight)
            self.fc.bias[:num_known].copy_(base_model.fc.bias)
    def forward_pre_mixup(self, x):
        return self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))
    def forward_post_mixup(self, h):
        x = torch.flatten(self.avgpool(self.layer4(self.layer3(h))), 1)
        return self.fc(x), x
    def forward(self, x):
        return self.forward_post_mixup(self.forward_pre_mixup(x))
