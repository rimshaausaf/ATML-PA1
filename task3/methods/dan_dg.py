import torch
import torch.nn as nn

def rbf_kernel(x, y):
    dist = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    median_dist = torch.median(dist[dist > 0]) if torch.sum(dist > 0) > 0 else torch.tensor(1.0).to(dist.device)
    bandwidths = [0.5 * median_dist, 1.0 * median_dist, 2.0 * median_dist]
    return sum(torch.exp(-dist / bw) for bw in bandwidths)

def mmd_loss(x, y):
    return rbf_kernel(x, x).mean() + rbf_kernel(y, y).mean() - 2 * rbf_kernel(x, y).mean()

class DANDGLoss(nn.Module):
    def __init__(self, lambda_dg=1.0):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.lambda_dg = lambda_dg

    def forward(self, logits_s, labels_s, feats_dict):
        cls_loss = self.cls_criterion(logits_s, labels_s)
        f_p = feats_dict['photo']
        f_a = feats_dict['art_painting']
        f_c = feats_dict['cartoon']
        mmd_pa = mmd_loss(f_p, f_a)
        mmd_pc = mmd_loss(f_p, f_c)
        mmd_ac = mmd_loss(f_a, f_c)
        avg_mmd = (mmd_pa + mmd_pc + mmd_ac) / 3.0
        total_loss = cls_loss + self.lambda_dg * avg_mmd
        return total_loss, cls_loss, avg_mmd
