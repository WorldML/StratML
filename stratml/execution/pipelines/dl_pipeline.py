"""
dl_pipeline.py
--------------
Phase 5 (DL) -- PyTorch training loop.

Architectures live in dl_architectures.py -- add new ones there.
This file owns: device selection, data loading, training loop,
early stopping, mixed precision, gradient clipping, TensorBoard.

Modality paths:
  tabular -- float32 features, unchanged
  vision  -- reshape flat -> (C, H, W), ImageNet normalization for pretrained
  text    -- int64 token IDs; BERT models receive attention_mask separately
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from stratml.execution.pipelines.dl_architectures import build_model
from stratml.execution.schemas import ExperimentConfig, DataSplit


# -- Device -------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -- Lazy Dataset -------------------------------------------------------------

class _TabularDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx : idx + 1]).squeeze(0)


# -- Result -------------------------------------------------------------------

@dataclass
class DLPipelineResult:
    model: nn.Module
    y_val_pred: np.ndarray
    train_curve: list[float]
    val_curve: list[float]
    runtime: float
    device_used: str = "cpu"
    epochs_run: int = 0
    early_stopped: bool = False
    best_epoch: int = 0
    model_state: dict = field(default_factory=dict)
    fit_count: int = 1
    eval_count: int = 1


# -- Main entry point ---------------------------------------------------------

def run_dl_pipeline(
    config: ExperimentConfig,
    data_split: DataSplit,
    tensorboard_log_dir: Optional[str] = None,
) -> DLPipelineResult:
    """
    Train a DL model and return predictions + training curves.

    Args:
        config:               ExperimentConfig with model_type="dl".
        data_split:           Pre-processed DataSplit (all numeric, no nulls).
        tensorboard_log_dir:  Optional path for TensorBoard SummaryWriter.
    """
    hp             = config.hyperparameters
    arch           = str(hp.get("architecture", "MLP"))
    task           = str(hp.get("task", "classification")).lower()
    modality       = str(hp.get("modality", "tabular")).lower()
    lr             = float(hp.get("learning_rate", 1e-3))
    weight_decay   = float(hp.get("weight_decay", 0.0))
    batch_size     = int(hp.get("batch_size", 32))
    epochs         = int(hp.get("epochs", 20))
    scheduler_type = str(hp.get("scheduler", "plateau")).lower()
    grad_clip      = float(hp.get("grad_clip", 1.0))
    mixed_prec     = bool(hp.get("mixed_precision", False))

    device  = _get_device()
    use_amp = mixed_prec and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # -- Numpy arrays ---------------------------------------------------------
    X_train_np = data_split.X_train.values.astype(np.float32)
    X_val_np   = data_split.X_val.values.astype(np.float32)
    input_dim  = X_train_np.shape[1]

    if task == "regression":
        y_train_np = data_split.y_train.values.astype(np.float32).reshape(-1, 1)
        y_val_np   = data_split.y_val.values.astype(np.float32).reshape(-1, 1)
        output_dim = 1
        criterion  = nn.MSELoss()
        label_map  = None
    else:
        classes    = sorted(data_split.y_train.unique())
        label_map  = {c: i for i, c in enumerate(classes)}
        y_train_np = data_split.y_train.map(label_map).values.astype(np.int64)
        y_val_np   = data_split.y_val.map(label_map).fillna(-1).values.astype(np.int64)
        output_dim = len(classes)
        criterion  = nn.CrossEntropyLoss()

    # -- Vision: reshape flat -> (C, H, W) and optionally normalize -----------
    _imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _arch_upper = arch.upper().replace("-", "").replace("_", "")
    _is_pretrained_vision = _arch_upper in {"RESNET18", "EFFICIENTNETB0", "MOBILENETV3"}
    _is_pretrained_text   = _arch_upper in {"DISTILBERT", "TINYBERT"}

    if modality == "vision":
        image_shape = tuple(hp.get("image_shape", (1, 28, 28)))
        C, H, W = image_shape
        X_train_np = X_train_np.reshape(-1, C, H, W)
        X_val_np   = X_val_np.reshape(-1, C, H, W)
        if _is_pretrained_vision and C == 1:
            # Pretrained models expect 3-channel input; replicate channel
            X_train_np = np.repeat(X_train_np, 3, axis=1)
            X_val_np   = np.repeat(X_val_np,   3, axis=1)
            C = 3
        if _is_pretrained_vision:
            # ImageNet normalization (values assumed in [0, 1])
            X_train_np = (X_train_np - _imagenet_mean[:, None, None]) / _imagenet_std[:, None, None]
            X_val_np   = (X_val_np   - _imagenet_mean[:, None, None]) / _imagenet_std[:, None, None]
        hp = {**hp, "image_shape": (C, H, W)}
        input_dim = C * H * W

    # -- Text: input is int64 token IDs ---------------------------------------
    _attention_mask_train: Optional[np.ndarray] = None
    _attention_mask_val:   Optional[np.ndarray] = None
    if modality == "text":
        X_train_np = X_train_np.astype(np.int64)
        X_val_np   = X_val_np.astype(np.int64)
        if _is_pretrained_text:
            _attention_mask_train = (X_train_np != 0).astype(np.int64)
            _attention_mask_val   = (X_val_np   != 0).astype(np.int64)

    # -- DataLoaders ----------------------------------------------------------
    _x_dtype = np.int64 if modality == "text" else np.float32
    train_loader = DataLoader(
        _TabularDataset(X_train_np.astype(_x_dtype), y_train_np),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    X_val_t = torch.tensor(X_val_np.astype(_x_dtype), device=device)
    y_val_t = torch.tensor(y_val_np, device=device)
    if task == "regression":
        y_val_t = y_val_t.reshape(-1, 1)
    _mask_val_t = (
        torch.tensor(_attention_mask_val, device=device)
        if _attention_mask_val is not None else None
    )

    # -- Model + optimiser + scheduler ----------------------------------------
    model     = build_model(arch, input_dim, output_dim, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler: Optional[object] = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -- TensorBoard ----------------------------------------------------------
    writer = None
    if tensorboard_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
        except Exception:
            pass

    # -- Training loop --------------------------------------------------------
    train_curve: list[float] = []
    val_curve:   list[float] = []
    best_val_loss    = float("inf")
    best_state_dict  = None
    best_epoch_idx   = 0
    patience_counter = 0
    early_stopped    = False
    patience = config.early_stopping_patience if config.early_stopping else max(5, epochs // 4)

    t0 = time.perf_counter()

    for epoch in tqdm(range(epochs), desc=f"  {arch}", unit="epoch", ncols=None,
                      dynamic_ncols=True, file=sys.stderr,
                      bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()

            def _forward(xb):
                if _is_pretrained_text:
                    # Build attention mask on-the-fly for this batch
                    mask = (xb != 0).long()
                    return model(xb, attention_mask=mask)
                return model(xb)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(_forward(xb), yb)
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(_forward(xb), yb)
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * len(xb)

        train_loss = round(epoch_loss / len(X_train_np), 6)
        train_curve.append(train_loss)

        model.eval()
        with torch.no_grad():
            if _is_pretrained_text and _mask_val_t is not None:
                val_out = model(X_val_t, attention_mask=_mask_val_t)
            else:
                val_out = model(X_val_t)
            val_loss = round(criterion(val_out, y_val_t).item(), 6)
        val_curve.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_idx   = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True
                break

    runtime = round(time.perf_counter() - t0, 4)

    if writer is not None:
        writer.close()

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # -- Predictions ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        if _is_pretrained_text and _mask_val_t is not None:
            out = model(X_val_t, attention_mask=_mask_val_t).cpu()
        else:
            out = model(X_val_t).cpu()

    if task == "regression":
        y_val_pred = out.squeeze(1).numpy()
    else:
        preds   = out.argmax(dim=1).numpy()
        inv_map = {i: c for c, i in label_map.items()}
        y_val_pred = np.array([inv_map[i] for i in preds])

    return DLPipelineResult(
        model=model.cpu(),
        y_val_pred=y_val_pred,
        train_curve=train_curve,
        val_curve=val_curve,
        runtime=runtime,
        device_used=str(device),
        epochs_run=len(train_curve),
        early_stopped=early_stopped,
        best_epoch=best_epoch_idx,
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
    )
