import torch.nn as nn

class SourceOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits_s, labels_s, **kwargs):
        loss = self.criterion(logits_s, labels_s)
        return loss, loss, 0.0
