# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import argparse
import os
import torch
import torch.nn as nn

from dataset import PretrainingDataset
from ecg_clip import ECGClip
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import make_optimizer_and_scheduler, random_obfuscate, recon_loss, tokenize_ecg


def run_epoch(model, loader, device, mask, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            ecg = tokenize_ecg(batch["ecg"].to(device))
            obf_ecg, clean_idx, obf_idx = random_obfuscate(ecg, mask)
            _, recon = model(obf_ecg, clean_idx, obf_idx)
            loss = recon_loss(recon, ecg, obf_idx)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG-CLIP stage 1: masked autoencoding pretraining")

    # model
    parser.add_argument("--token_dim",       type=int,   default=500)
    parser.add_argument("--hidden_dim",      type=int,   default=512)
    parser.add_argument("--embedding_dim",   type=int,   default=512)
    parser.add_argument("--enc_depth",       type=int,   default=12)
    parser.add_argument("--dec_depth",       type=int,   default=6)
    parser.add_argument("--num_heads",       type=int,   default=8)
    parser.add_argument("--dropout",         type=float, default=0.0)
    parser.add_argument("--contrastive_dim", type=int,   default=512)

    # training
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--beta1",         type=float, default=0.9)
    parser.add_argument("--beta2",         type=float, default=0.98)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--max_epochs",    type=int,   default=1600)
    parser.add_argument("--warmup_epochs", type=int,   default=40)
    parser.add_argument("--mask",          type=float, default=0.75)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--gpu_id",        type=int,   default=0)
    parser.add_argument("--output_dir",    type=str,   default="output")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        PretrainingDataset(),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PretrainingDataset(),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = ECGClip(
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
    ).to(device)

    niter_per_epoch = len(train_loader)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args.lr, args.beta1, args.beta2, args.weight_decay,
        args.max_epochs, niter_per_epoch, args.warmup_epochs,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, args.mask, optimizer, scheduler)
        val_loss   = run_epoch(model, val_loader,   device, args.mask)

        print(f"Epoch {epoch:3d}/{args.max_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(args)},
                f"{args.output_dir}/stage1_best.pt",
            )

    print(f"\nBest val loss: {best_val_loss:.4f}")

