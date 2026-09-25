import os
import csv
import argparse
import copy
import torch
import torch.optim as optim
import numpy as np
from shared.pacs_protocol import get_pacs_dataloaders
from shared.pacs import InfiniteDomainIterator
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.methods.dan_dg import DANDGLoss
from task3.methods.sam import SAMOptimizer
from task2.evaluation.metrics import run_evaluation

def train_dg(method, lambda_dg=1.0, rho=0.05, seed=6304):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)    
    source_train, _, source_val, _ = get_pacs_dataloaders(seed=seed)
    iter_photo = InfiniteDomainIterator(source_train["photo"])
    iter_art = InfiniteDomainIterator(source_train["art_painting"])
    iter_cartoon = InfiniteDomainIterator(source_train["cartoon"])
    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead(in_features=512, num_classes=7).to(device)
    trainable_params = list(backbone.parameters()) + list(classifier.parameters())
    base_optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    
    if method == "sam":
        sam_opt = SAMOptimizer(trainable_params, base_optimizer, rho=rho)
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif method == "dan_dg":
        criterion = DANDGLoss(lambda_dg=lambda_dg).to(device)
    max_epochs = 30
    patience = 5
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    best_state = None
    batches_per_epoch = max(len(source_train["photo"]), len(source_train["art_painting"]), len(source_train["cartoon"]))
    
    os.makedirs("task3/results", exist_ok=True)
    if method == "dan_dg":
        run_id = f"dan_dg_lambda{lambda_dg}"
    else:
        run_id = f"sam_rho{rho}"
    log_name = f"task3/results/{run_id}_losses.csv"
    with open(log_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "class_loss", "mmd_penalty", "photo_f1", "art_f1", "cartoon_f1", "mean_f1", "worst_f1"])
        
    print(f"Starting Task 3 [{run_id.upper()}] training on {device}...")
    
    for epoch in range(max_epochs):
        backbone.train()
        classifier.train()
        epoch_cls_loss = 0.0
        epoch_mmd_loss = 0.0
        for _ in range(batches_per_epoch):
            p_img, p_lbl = iter_photo.next()
            a_img, a_lbl = iter_art.next()
            c_img, c_lbl = iter_cartoon.next()
            source_imgs = torch.cat([p_img, a_img, c_img], dim=0).to(device)
            source_lbls = torch.cat([p_lbl, a_lbl, c_lbl], dim=0).to(device)
            if method == "dan_dg":
                base_optimizer.zero_grad()
                f_p = backbone(p_img.to(device))
                f_a = backbone(a_img.to(device))
                f_c = backbone(c_img.to(device))
                all_feats = torch.cat([f_p, f_a, f_c], dim=0)
                all_logits = classifier(all_feats)
                feats_dict = {'photo': f_p, 'art_painting': f_a, 'cartoon': f_c}
                loss, c_loss, mmd_loss_val = criterion(all_logits, source_lbls, feats_dict)
                loss.backward()
                base_optimizer.step()
                epoch_cls_loss += c_loss.item()
                epoch_mmd_loss += mmd_loss_val.item()
                
            elif method == "sam":
                sam_opt.zero_grad()
                feats = backbone(source_imgs)
                logits = classifier(feats)
                loss = criterion(logits, source_lbls)
                loss.backward()
                sam_opt.first_step()
                backbone.train()
                feats_pert = backbone(source_imgs)
                logits_pert = classifier(feats_pert)
                loss_pert = criterion(logits_pert, source_lbls)
                sam_opt.zero_grad()
                loss_pert.backward()
                sam_opt.second_step()
                epoch_cls_loss += loss.item()
                epoch_mmd_loss = 0.0
                
        val_f1s = {}
        for domain, loader in source_val.items():
            _, f1, _, _, _ = run_evaluation(backbone, classifier, loader, device)
            val_f1s[domain] = f1
            
        mean_val_f1 = np.mean(list(val_f1s.values()))
        worst_val_f1 = np.min(list(val_f1s.values()))
        
        with open(log_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_cls_loss / batches_per_epoch,
                epoch_mmd_loss / batches_per_epoch,
                val_f1s['photo'], val_f1s['art_painting'], val_f1s['cartoon'],
                mean_val_f1, worst_val_f1
            ])
            
        print(f"Epoch {epoch+1:02d} | Cls: {epoch_cls_loss/batches_per_epoch:.4f} | MMD: {epoch_mmd_loss/batches_per_epoch:.4f} | Mean F1: {mean_val_f1:.2f}% | Worst F1: {worst_val_f1:.2f}%")
        
        if mean_val_f1 > best_macro_f1:
            best_macro_f1 = mean_val_f1
            epochs_no_improve = 0
            best_state = {
                'backbone': copy.deepcopy(backbone.state_dict()),
                'classifier': copy.deepcopy(classifier.state_dict())
            }
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    save_path = f"task3/results/{run_id}_best.pth"
    torch.save(best_state, save_path)
    print(f"Saved best model checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["dan_dg", "sam"])
    parser.add_argument("--lambda_dg", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.05)
    args = parser.parse_args()
    train_dg(args.method, args.lambda_dg, args.rho)
