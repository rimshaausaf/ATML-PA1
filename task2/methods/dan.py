import torch
import torch.nn as nn

class DANLoss(nn.Module):
    def __init__(self, lambda_mmd=1.0):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.cls_criterion = nn.CrossEntropyLoss()

    def rbf_kernel_mmd(self, source, target):
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if torch.sum(L2_distance) == 0:
            median_sq = 1.0
        else:
            dist_flat = L2_distance.view(-1)
            median_sq = torch.median(dist_flat[dist_flat > 0]).item()
            if median_sq == 0:
                median_sq = 1.0
        bandwidths = [0.5 * median_sq, 1.0 * median_sq, 2.0 * median_sq]
        kernel_val = sum([torch.exp(-L2_distance / bw) for bw in bandwidths])
        batch_size_s = source.size(0)
        XX = kernel_val[:batch_size_s, :batch_size_s]
        YY = kernel_val[batch_size_s:, batch_size_s:]
        XY = kernel_val[:batch_size_s, batch_size_s:]
        YX = kernel_val[batch_size_s:, :batch_size_s]
        mmd_loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return mmd_loss

    def forward(self, logits_s, labels_s, feats_s, feats_t):
        cls_loss = self.cls_criterion(logits_s, labels_s)
        mmd_loss = self.rbf_kernel_mmd(feats_s, feats_t)
        total_loss = cls_loss + self.lambda_mmd * mmd_loss
        return total_loss, cls_loss, mmd_loss
