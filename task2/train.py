import os
import csv
import argparse
import torch
import torch.optim as optim
import numpy as np
import copy

from shared.pacs_protocol import get_pacs_dataloaders
from shared.pacs import InfiniteDomainIterator
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator
from task2.methods.source_only import SourceOnlyLoss
from task2.methods.dan import DANLoss
from task2.methods.dann import DANNLoss
from task2.methods.cdan import CDANLoss
from task2.evaluation.metrics import run_evaluation

def train(method, max_alpha=1.0, seed=6304):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    source_train, target_train, source_val, _ = get_pacs_dataloaders(seed=seed)
    
    iter_photo = InfiniteDomainIterator(source_train["photo"])
    iter_art = InfiniteDomainIterator(source_train["art_painting"])
    iter_cartoon = InfiniteDomainIterator(source_train["cartoon"])
    
    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead(in_features=512, num_classes=7).to(device)
    
    trainable_params = list(backbone.parameters()) + list(classifier.parameters())
    
    if method == "dann":
        domain_discriminator = DomainDiscriminator(in_features=512).to(device)
        trainable_params += list(domain_discriminator.parameters())
    elif method == "cdan":
        domain_discriminator = DomainDiscriminator(in_features=512 * 7).to(device)
        trainable_params += list(domain_discriminator.parameters())
    else:
        domain_discriminator = None
        
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    
    if method == "source_only":
        criterion = SourceOnlyLoss().to(device)
    elif method == "dan":
        criterion = DANLoss(lambda_mmd=1.0).to(device)
    elif method == "dann":
        criterion = DANNLoss(domain_discriminator).to(device)
    elif method == "cdan":
        criterion = CDANLoss(domain_discriminator).to(device)

    max_epochs = 30
    patience = 5
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    best_state = None
    
    batches_per_epoch = len(target_train)
    total_steps = max_epochs * batches_per_epoch
    global_step = 0
    
    os.makedirs("task2/results", exist_ok=True)
    log_name = f"task2/results/{method}_alpha{max_alpha}_losses.csv"
    with open(log_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "class_loss", "domain_loss", "photo_f1", "art_f1", "cartoon_f1", "mean_f1"])
    
    print(f"Starting {method.upper()} training on {device} (Max Alpha: {max_alpha})...")
    
    for epoch in range(max_epochs):
        backbone.train() 
        classifier.train()
        if domain_discriminator:
            domain_discriminator.train()
            
        epoch_cls_loss = 0.0
        epoch_dom_loss = 0.0
        
        for target_imgs, _ in target_train:
            target_imgs = target_imgs.to(device)
            
            p_img, p_lbl = iter_photo.next()
            a_img, a_lbl = iter_art.next()
            c_img, c_lbl = iter_cartoon.next()
            
            source_imgs = torch.cat([p_img, a_img, c_img], dim=0).to(device)
            source_lbls = torch.cat([p_lbl, a_lbl, c_lbl], dim=0).to(device)
            
            p = float(global_step) / total_steps
            alpha = max_alpha * ((2.0 / (1.0 + np.exp(-10 * p))) - 1.0)
            
            optimizer.zero_grad()
            
            feats_s = backbone(source_imgs)
            logits_s = classifier(feats_s)
            
            if method == "source_only":
                loss, c_loss, d_loss = criterion(logits_s, source_lbls)
            else:
                feats_t = backbone(target_imgs)
                logits_t = classifier(feats_t)
                
                if method == "dan":
                    loss, c_loss, d_loss = criterion(logits_s, source_lbls, feats_s, feats_t)
                elif method == "dann":
                    loss, c_loss, d_loss = criterion(logits_s, source_lbls, feats_s, feats_t, alpha)
                elif method == "cdan":
                    loss, c_loss, d_loss = criterion(logits_s, source_lbls, feats_s, logits_t, feats_t, alpha)
                
            loss.backward()
            optimizer.step()
            
            epoch_cls_loss += c_loss.item()
            epoch_dom_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            global_step += 1
            
        val_f1s = {}
        for domain, loader in source_val.items():
            _, f1, _, _, _ = run_evaluation(backbone, classifier, loader, device)
            val_f1s[domain] = f1
            
        mean_val_f1 = np.mean(list(val_f1s.values()))
        
        with open(log_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                epoch_cls_loss / batches_per_epoch, 
                epoch_dom_loss / batches_per_epoch,
                val_f1s['photo'], val_f1s['art_painting'], val_f1s['cartoon'], mean_val_f1
            ])
            
        print(f"Epoch {epoch+1:02d} | Cls Loss: {epoch_cls_loss/batches_per_epoch:.4f} | Dom Loss: {epoch_dom_loss/batches_per_epoch:.4f} | Mean Source F1: {mean_val_f1:.2f}%")
        
        if mean_val_f1 > best_macro_f1:
            best_macro_f1 = mean_val_f1
            epochs_no_improve = 0
            best_state = {
                'backbone': copy.deepcopy(backbone.state_dict()),
                'classifier': copy.deepcopy(classifier.state_dict())
            }
            if domain_discriminator:
                best_state['domain_discriminator'] = copy.deepcopy(domain_discriminator.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    save_path = f"task2/results/{method}_alpha{max_alpha}_best.pth"
    torch.save(best_state, save_path)
    print(f"Saved best checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["source_only", "dan", "dann", "cdan"])
    parser.add_argument("--max_alpha", type=float, default=1.0)
    args = parser.parse_args()
    train(args.method, args.max_alpha)
