# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import argparse
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn

from dataset import FinetuneDataset
from ecg_clip import ECGClip, ECGClipFinetuning
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import tokenize_ecg


def build_optimizer_scheduler(
    model: ECGClipFinetuning,
    lr: float,
    layer_decay: float,
    max_epochs: int,
    niter_per_epoch: int,
    warmup_iters: int = 100,
):
    linear_layer = model.linear_layer
    encoder_FC = model.model.encoder_FC
    norm = model.model.encoder_transformer.norm
    transformer_layers = model.model.encoder_transformer.layers
    linear_projection = model.model.linear_projection
    n_layers = model.model.enc_depth + 2

    parameters = [
        {"params": linear_layer.parameters(),  "lr": lr},
        {"params": encoder_FC.parameters(),    "lr": lr * layer_decay},
        {"params": norm.parameters(),          "lr": lr * layer_decay},
    ]
    for depth in range(2, n_layers):
        parameters.append({
            "params": transformer_layers[n_layers - depth - 1].parameters(),
            "lr": lr * (layer_decay ** depth),
        })
    parameters.append({
        "params": linear_projection.parameters(),
        "lr": lr * (layer_decay ** n_layers),
    })

    optimizer = AdamW(parameters, betas=(0.9, 0.999), weight_decay=0.05)
    training_steps = max_epochs * niter_per_epoch
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, training_steps - warmup_iters))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
    return optimizer, scheduler


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            ecg = tokenize_ecg(batch["ecg"].to(device))
            labels = batch["label"].to(device)

            logits = model(ecg).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().float().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader)
    auroc = roc_auc_score(all_labels, all_logits)
    return avg_loss, auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ECG-CLIP for binary classification")

    parser.add_argument("--checkpoint",    type=str,   default=None, help="Path to ECGClip pretrained weights")
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--layer_decay",   type=float, default=0.8)
    parser.add_argument("--warmup_iters",  type=int,   default=100)
    parser.add_argument("--max_epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--num_workers",   type=int,   default=4)

    # misc
    parser.add_argument("--gpu_id",        type=int,   default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- dataloaders ---
    train_loader = DataLoader(
        FinetuneDataset(),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FinetuneDataset(),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        FinetuneDataset(),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # --- model ---
    backbone = ECGClip(
        token_dim=500,
        hidden_dim=512,
        embedding_dim=512,
        enc_depth=12,
        dec_depth=6,
        num_heads=8,
        dropout=0.0,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    model = ECGClipFinetuning(backbone, output_dim=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    niter_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_scheduler(
        model, args.lr, args.layer_decay, args.max_epochs, niter_per_epoch, args.warmup_iters
    )

    # --- training loop ---
    best_val_auroc = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_auroc = run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        val_loss,   val_auroc   = run_epoch(model, val_loader,   device, criterion)

        print(
            f"Epoch {epoch:3d}/{args.max_epochs} | "
            f"train loss {train_loss:.4f}  auroc {train_auroc:.4f} | "
            f"val loss {val_loss:.4f}  auroc {val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- final test with best checkpoint ---
    model.load_state_dict(best_state)
    _, test_auroc = run_epoch(model, test_loader, device, criterion)

    print(
        f"\nBest epoch {best_epoch} | "
        f"val AUROC {best_val_auroc:.4f} | "
        f"test AUROC {test_auroc:.4f}"
    )

