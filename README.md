# ECG-CLIP

Transformer-based ECG encoder pretrained with masked autoencoding and ECG-text contrastive learning (CLIP). Finetuned for binary classification on downstream clinical tasks.

## ECG format

- **Sample rate:** 500 Hz
- **Duration:** 10 seconds (5000 timepoints per lead)
- **Leads:** I, II, V1, V2, V3, V4, V5, V6 (8 leads)
- **Tensor shape:** `(8, 5000)` float32

Internally, each ECG is tokenized into 80 non-overlapping 500-sample windows across all leads: `(8, 5000) → (80, 500)`.

## Files

| File | Purpose |
|------|---------|
| `ecg_clip.py` | Model definitions (`ECGClip`, `ECGClipFinetuning`, `TextEncoder`) |
| `utils.py` | Tokenization, masking, reconstruction loss, CLIP loss, optimizer/scheduler |
| `dataset.py` | Dataset stubs — **implement these for your data** |
| `pretrain_stage1.py` | Stage 1 pretraining: masked autoencoding only |
| `pretrain_stage2.py` | Stage 2 pretraining: masked autoencoding + CLIP loss |
| `finetune.py` | Supervised finetuning with layer-wise LR decay |
| `downstream_tasks.py` | Build evaluation CSVs from MIMIC-IV data |

## Pretraining

Pretraining is two-stage. The hyperparameters in each script reflect the values used for the published model — they generally do not need to be changed.

**Stage 1 — masked autoencoding**

```bash
python pretrain_stage1.py --output_dir output/
```

Key defaults: 1600 epochs, batch size 1024, mask ratio 0.75, lr 1e-4, warmup 40 epochs.

**Stage 2 — contrastive + reconstruction**

Warm-starts from the stage 1 checkpoint. Requires ECG-report pairs: `dataset.py` must return `{"ecg": ..., "text": "..."}`.

```bash
python pretrain_stage2.py \
    --stage1_checkpoint output/stage1_best.pt \
    --output_dir output/
```

Key defaults: 100 epochs, batch size 1024, mask ratio 0.75, λ=0.99 (reconstruction weight), lr 1e-4, warmup 20 epochs. The text encoder is `emilyalsentzer/Bio_ClinicalBERT` with all layers frozen except the last encoder layer and the pooler.

## Finetuning

Loads a pretrained checkpoint and trains a linear head for binary classification. **`lr` and `layer_decay` must be tuned** for each downstream task.

```bash
python finetune.py \
    --checkpoint output/stage2_best.pt \
    --lr 1e-5 \
    --layer_decay 0.8
```

The optimizer applies layer-wise LR decay: the linear head trains at `lr`, and each encoder layer scales by an additional factor of `layer_decay` going deeper. Metric is AUROC; best val checkpoint is evaluated on test.

## Implementing `dataset.py`

`PretrainingDataset` and `FinetuneDataset` are stubs. Replace their `__getitem__` with real data loading — the dict keys must match what the training scripts expect:

- **Pretraining:** `{"ecg": Tensor(8, 5000), "text": str}`
- **Finetuning:** `{"ecg": Tensor(8, 5000), "label": float scalar Tensor}`

## Downstream tasks

`downstream_tasks.py` builds labeled CSVs from MIMIC-IV for eight tasks:

| Task | Label source |
|------|-------------|
| Acute MI | ECG report keywords |
| AF prediction | Index NSR ECG before first AF/AFL |
| Cardiac amyloidosis (CA) | ICD codes, ECG within 180 days prior |
| HCM | ICD codes |
| ED 30-day mortality | MIMIC-IV-ED MDS |
| Surgery 30-day mortality | ICD procedure codes + death records |
| CKD (3-year risk) | Creatinine labs, CKD-EPI eGFR ≤ 45 |
| DM (3-year risk) | HbA1c ≥ 6.5 |

```bash
python downstream_tasks.py \
    --mimic_dir /path/to/mimic-iv \
    --mimic_ecg_dir /path/to/mimic-iv-ecg \
    --mimic_ext_mds_ed_dir /path/to/mimic-iv-ed-mds \
    --hcup_icd9_pth /path/to/hcup_icd9.csv \
    --hcup_icd10_pth /path/to/hcup_icd10.csv \
    --output_dir /path/to/output_dir
```

Outputs one CSV per task. Each row has `subject_id`, `study_id`, `path`, and a binary label column.
