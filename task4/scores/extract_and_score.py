import torch
import numpy as np
import torch.nn.functional as F
def extract_features(model, loader, device):
    model.eval()
    all_feats, all_logits, all_targets = [], [], []
    feats_out = []
    def hook(module, input, output):
        feats_out.append(input[0].detach())
    handle = model.fc.register_forward_hook(hook)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_logits.append(logits.cpu())
            all_feats.append(feats_out.pop().cpu())
            all_targets.append(targets)
    handle.remove()
    return torch.cat(all_feats), torch.cat(all_logits), torch.cat(all_targets)
def compute_mahalanobis_params(train_feats, train_targets):
    num_classes = 10
    class_means, variances = [], []
    for c in range(num_classes):
        idx = (train_targets == c)
        c_feats = train_feats[idx]
        class_means.append(c_feats.mean(dim=0))
        variances.append(c_feats.var(dim=0, unbiased=True))
    shared_var = torch.stack(variances).mean(dim=0) + 1e-6
    return torch.stack(class_means), shared_var
def get_scores(logits, features, class_means=None, shared_var=None):
    probs = F.softmax(logits, dim=1)
    msp = 1.0 - probs.max(dim=1)[0]
    mls = -logits.max(dim=1)[0]
    energy = -torch.logsumexp(logits, dim=1)
    mahalanobis = None
    if class_means is not None and shared_var is not None:
        distances = []
        for c in range(10):
            diff = features - class_means[c]
            dist = (diff.pow(2) / shared_var).sum(dim=1)
            distances.append(dist)
        mahalanobis = torch.stack(distances, dim=1).min(dim=1)[0]
    return msp, mls, energy, mahalanobis
