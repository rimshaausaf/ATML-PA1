import torch
import torch.nn as nn
import torch.nn.functional as F

class CDANLoss(nn.Module):
    def __init__(self, domain_discriminator):
        super().__init__()
        self.domain_discriminator = domain_discriminator
        self.cls_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(self, logits_s, labels_s, feats_s, logits_t, feats_t, alpha):
        cls_loss = self.cls_criterion(logits_s, labels_s)
        prob_s = F.softmax(logits_s, dim=1)
        prob_t = F.softmax(logits_t, dim=1)
        batch_size_s = feats_s.size(0)
        batch_size_t = feats_t.size(0)
        op_s = torch.bmm(feats_s.unsqueeze(2), prob_s.unsqueeze(1)).view(batch_size_s, -1)
        op_t = torch.bmm(feats_t.unsqueeze(2), prob_t.unsqueeze(1)).view(batch_size_t, -1)
        domain_labels_s = torch.zeros(batch_size_s, dtype=torch.long, device=feats_s.device)
        domain_labels_t = torch.ones(batch_size_t, dtype=torch.long, device=feats_t.device)
        logits_d_s = self.domain_discriminator(op_s, alpha)
        logits_d_t = self.domain_discriminator(op_t, alpha)
        domain_loss_s = self.domain_criterion(logits_d_s, domain_labels_s)
        domain_loss_t = self.domain_criterion(logits_d_t, domain_labels_t)
        domain_loss = (domain_loss_s + domain_loss_t) / 2.0
        total_loss = cls_loss + domain_loss
        return total_loss, cls_loss, domain_loss
