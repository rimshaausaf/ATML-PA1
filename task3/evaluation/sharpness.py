import torch
import torch.nn as nn
import numpy as np

def compute_sharpness_proxy(backbone, classifier, source_val_loaders, device, seed=6304, rho=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected_imgs = []
    selected_lbls = []
    for domain in ['photo', 'art_painting', 'cartoon']:
        loader = source_val_loaders[domain]
        imgs_d, lbls_d = [], []
        for imgs, lbls in loader:
            imgs_d.append(imgs)
            lbls_d.append(lbls)
        all_imgs = torch.cat(imgs_d, dim=0)
        all_lbls = torch.cat(lbls_d, dim=0)
        
        idx = np.random.choice(len(all_imgs), 32, replace=False)
        selected_imgs.append(all_imgs[idx])
        selected_lbls.append(all_lbls[idx])
        
    eval_imgs = torch.cat(selected_imgs, dim=0).to(device)
    eval_lbls = torch.cat(selected_lbls, dim=0).to(device)
    backbone.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    feats = backbone(eval_imgs)
    logits = classifier(feats)
    base_loss = criterion(logits, eval_lbls)
    params = list(backbone.parameters()) + list(classifier.parameters())
    backbone.zero_grad()
    classifier.zero_grad()
    base_loss.backward()
    norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for p in params if p.grad is not None]),
        p=2
    )
    scale = rho / (norm + 1e-12)
    
    saved_params = {}
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                saved_params[p] = p.data.clone()
                p.add_(p.grad * scale)
    with torch.no_grad():
        p_feats = backbone(eval_imgs)
        p_logits = classifier(p_feats)
        perturbed_loss = criterion(p_logits, eval_lbls)
    with torch.no_grad():
        for p in params:
            if p in saved_params:
                p.data = saved_params[p]
                
    delta_sharp = (perturbed_loss - base_loss).item()
    return delta_sharp
