"""
Train Bacformer for a binary isolation-source pair from pytorch (.pt) files.

Paths and filenames follow ``prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py``:
``training_<slug1>_<slug2>/{train,validate,evaluate}/`` and ``binary_<pair_slug>_with_split.csv``.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from bacformer.modeling.trainer import BacformerLargeTrainer
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    TrainingArguments,
)

from predict_kleb_by_bacformer.pp.isolation_source_cli_parsing import (
    sanitize_pair_name,
    slugify_isolation_source_token,
)

PROCESSED_BASE_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed"
)


############################################################## PyTorchFileDataset ##############################################################
class PyTorchFileDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads pytorch (.pt) files for isolation-source pair training."""

    def __init__(
        self,
        file_paths: list[Path],
        label_column: str,
    ):
        self.file_paths = file_paths
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        if self.label_column not in data or pd.isna(data[self.label_column]):
            raise ValueError(
                f"Sample {data.get('Sample', 'unknown')} has no label for {self.label_column}"
            )

        label_val = data[self.label_column]
        if isinstance(label_val, torch.Tensor):
            label_val = label_val.item()
        elif hasattr(label_val, "item"):
            label_val = label_val.item()
        label_val = int(label_val)

        prot_embeddings = data.get("prot_embeddings", data.get("protein_embeddings"))
        if prot_embeddings is None:
            raise KeyError(
                f"Sample {data.get('Sample', 'unknown')} is missing 'prot_embeddings'/'protein_embeddings'."
            )

        if prot_embeddings.dim() == 2:
            prot_embeddings = prot_embeddings.unsqueeze(0)

        seq_len = prot_embeddings.shape[1]
        sample: dict[str, torch.Tensor] = {
            "protein_embeddings": prot_embeddings,
            "labels": torch.tensor(label_val, dtype=torch.float32),
        }

        am = data["attention_mask"] if "attention_mask" in data else None
        if am is not None and am.dim() == 1:
            am = am.unsqueeze(0)
        sample["attention_mask"] = am if am is not None else torch.ones(1, seq_len, dtype=torch.float32)
        contig_src = data.get("contig_idx", data.get("contig_ids", data.get("token_type_ids")))
        if contig_src is not None:
            sample["contig_ids"] = (
                contig_src.unsqueeze(0) if contig_src.dim() == 1 else contig_src
            )
        else:
            sample["contig_ids"] = torch.zeros(1, seq_len, dtype=torch.long)

        return sample


############################################################## Compute metrics ##############################################################


def compute_metrics_binary_genome_pred(
    preds: EvalPrediction, ignore_index: int = -100, prefix: str = "eval"
):
    """Compute metrics for a single-logit binary classifier."""
    with torch.no_grad():
        logits = torch.tensor(preds.predictions).flatten()
        labels = torch.tensor(preds.label_ids).flatten().long()
        if (labels == ignore_index).any():
            keep = labels != ignore_index
            logits, labels = logits[keep], labels[keep]
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

        y_true = labels.cpu().numpy()
        y_prob = prob.cpu().numpy()
        y_pred = pred.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        try:
            auroc_val = roc_auc_score(y_true, y_prob)
        except Exception:  # noqa
            auroc_val = float("nan")
        try:
            auprc = average_precision_score(y_true, y_prob)
        except Exception:  # noqa
            auprc = float("nan")
        f1 = f1_score(y_true, y_pred, average="binary")

    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_auroc": auroc_val,
        f"{prefix}_auprc": auprc,
        f"{prefix}_f1": f1,
        f"{prefix}_nr_samples": len(y_true),
    }


############################################################## Main run function ##############################################################


def run(
    model_name_or_path: str,
    train_data_dir: str,
    val_data_dir: str,
    output_dir: str,
    sheet_path: str,
    label_column: str,
    pt_suffix: str,
    lr: float = 0.00015,
    batch_size: int = 1,
    grad_accumulation_steps: int = 8,
    max_n_proteins: int = 6000,
    freeze_encoder: bool = False,
    logging_steps: int = 10,
    n_samples: int = 10000,
    seed: int = 1,
    early_stopping_patience: int = 10,
    eval_steps: int = 150,
    num_workers: int = 16,
    warmup_proportion: float = 0.1,
    max_steps: int = 20000,
):
    """Fine-tune Bacformer on a pair-specific isolation-source label using pytorch (.pt) files."""
    print(f"Loading model from: {model_name_or_path}")
    print(f"Predicting {label_column}")
    print(f"Training set (PT): {train_data_dir}")
    print(f"Validation set (PT): {val_data_dir}")
    print(f"PT filename suffix: {pt_suffix}")
    print(f"n_samples = {n_samples}")
    print(f"Freezing encoder: {freeze_encoder}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Number of workers: {num_workers}")
    print(f"Eval steps: {eval_steps}")
    print(f"Grad accumulation steps: {grad_accumulation_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Warmup proportion: {warmup_proportion}")
    print(f"Output directory: {output_dir}")
    print(f"Sheet: {sheet_path}")
    print("------------------------------------------------\n")

    if not sheet_path:
        raise ValueError("sheet_path must be provided (binary_<pair_slug>_with_split.csv).")
    if not os.path.exists(sheet_path):
        raise FileNotFoundError(f"Sheet not found at {sheet_path}")

    df = pd.read_csv(sheet_path)
    if "train_val_eval" not in df.columns:
        raise ValueError(
            "Sheet must contain 'train_val_eval' column. "
            "Run prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py first."
        )
    if "Sample" not in df.columns:
        if "sample_accession" in df.columns:
            df["Sample"] = df["sample_accession"].astype(str)
        elif "phenotype-BioSample_ID" in df.columns:
            df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
        else:
            raise ValueError(
                "Sheet must contain 'Sample', 'sample_accession', or 'phenotype-BioSample_ID'."
            )
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in sheet.")

    counts = (
        df.loc[df[label_column].notna()]
        .groupby("train_val_eval")["Sample"]
        .nunique()
        .to_dict()
    )
    train_count = counts.get("train", 0)
    val_count = counts.get("validate", 0)
    eval_count = counts.get("evaluate", 0)
    print(
        f"Samples with non-missing '{label_column}' - train: {train_count}, "
        f"validate: {val_count}, evaluate: {eval_count}"
    )

    train_dir = Path(train_data_dir)
    val_dir = Path(val_data_dir)

    def build_file_list(split_name: str) -> list[Path]:
        ids = (
            df[(df["train_val_eval"] == split_name) & df[label_column].notna()]["Sample"]
            .astype(str)
            .unique()
        )
        base_dir = train_dir if split_name == "train" else val_dir
        if split_name not in ("train", "validate"):
            return []
        files = []
        for sid in ids:
            path = base_dir / f"{sid}{pt_suffix}"
            if path.exists():
                files.append(path)
            else:
                print(f"WARNING: Expected pytorch (.pt) file missing for Sample {sid}: {path}")
        return files

    if not train_dir.exists():
        raise FileNotFoundError(f"Train data directory {train_data_dir} does not exist")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation data directory {val_data_dir} does not exist")

    if n_samples == 10:
        print("Using dummy test mode with 10 samples.")
        train_ids = (
            df[(df["train_val_eval"] == "train") & df[label_column].notna()]["Sample"]
            .astype(str)
            .unique()[:10]
        )
        train_files = [
            train_dir / f"{sid}{pt_suffix}"
            for sid in train_ids
            if (train_dir / f"{sid}{pt_suffix}").exists()
        ]
        val_files = train_files
        eval_strategy = "epoch"
        use_epochs = True
        num_train_epochs = 100
    else:
        print("Full set mode: loading pytorch (.pt) files via PyTorchFileDataset")
        train_files = build_file_list("train")
        val_files = build_file_list("validate")
        eval_strategy = "steps"
        use_epochs = False

    if not train_files:
        raise RuntimeError(
            f"No pytorch (.pt) files found for training (check sheet, {label_column}, and {train_data_dir})"
        )

    print(f"Number of train files (samples with '{label_column}'): {len(train_files)}")
    print(f"Number of validation files: {len(val_files)}")

    train_dataset = PyTorchFileDataset(train_files, label_column=label_column)
    val_dataset = PyTorchFileDataset(val_files, label_column=label_column)

    try:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        emb = sample["protein_embeddings"]
        print(f"protein_embeddings shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        print(f"labels shape: {sample['labels'].shape}, value: {sample['labels'].item()}")
    except Exception as e:
        print(f"WARNING: Could not inspect sample: {e}")

    bacformer_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        problem_type="binary_classification",
        return_dict=True,
        trust_remote_code=True,
    ).to(torch.bfloat16)

    if freeze_encoder:
        for param in bacformer_model.bacformer.parameters():
            param.requires_grad = False

    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    os.makedirs(output_dir, exist_ok=True)

    def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        prot_list = [s["protein_embeddings"].squeeze(0) for s in samples]
        am_list = [s["attention_mask"].squeeze(0) for s in samples]
        contig_list = [s["contig_ids"].squeeze(0) for s in samples]
        return {
            "protein_embeddings": pad_sequence(prot_list, batch_first=True, padding_value=0.0),
            "labels": torch.stack([s["labels"] for s in samples], dim=0),
            "attention_mask": pad_sequence(am_list, batch_first=True, padding_value=0.0),
            "contig_ids": pad_sequence(contig_list, batch_first=True, padding_value=0),
        }

    training_args_dict = {
        "output_dir": output_dir,
        "eval_strategy": eval_strategy,
        "save_strategy": eval_strategy,
        "save_total_limit": 1,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "seed": seed,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": bool(num_workers > 0),
        "bf16": bool(torch.cuda.is_available()),
        "metric_for_best_model": "eval_auroc",
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
    }

    if use_epochs:
        total_batches = len(train_files) // batch_size
        steps_per_epoch = max(1, total_batches // grad_accumulation_steps)
        calculated_max_steps = max(1, steps_per_epoch * num_train_epochs)
        training_args_dict["max_steps"] = calculated_max_steps
        training_args_dict["num_train_epochs"] = num_train_epochs
        print(f"num_train_epochs: {num_train_epochs}, calculated max_steps: {calculated_max_steps}")
    else:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 in full dataset mode. Pass --max-steps.")
        training_args_dict["max_steps"] = max_steps
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["save_steps"] = eval_steps
        warmup_steps = int(max_steps * warmup_proportion)
        training_args_dict["warmup_steps"] = warmup_steps
        training_args_dict["lr_scheduler_type"] = "linear"
        print(f"Warmup steps: {warmup_steps}, max_steps: {max_steps}, eval_steps: {eval_steps}")

    training_args = TrainingArguments(**training_args_dict)

    trainer = BacformerLargeTrainer(
        model=bacformer_model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics_binary_genome_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune Bacformer for an isolation-source pair from split .pt files "
        "(same layout as prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py)."
    )
    p.add_argument(
        "--isolation-sources",
        nargs=2,
        metavar=("TOKEN1", "TOKEN2"),
        required=True,
        help="Two isolation-source CLI tokens (same as the prepare script).",
    )
    p.add_argument(
        "--processed-base-dir",
        type=str,
        default=str(PROCESSED_BASE_DIR_DEFAULT),
        help="Directory containing training_<slug1>_<slug2>/ (default matches prepare script).",
    )
    p.add_argument(
        "--train-data-dir",
        type=str,
        default=None,
        help="Override train split directory (default: <base>/training_<slug1>_<slug2>/train).",
    )
    p.add_argument(
        "--val-data-dir",
        type=str,
        default=None,
        help="Override validation split directory (default: .../validate).",
    )
    p.add_argument(
        "--sheet-path",
        type=str,
        default=None,
        help="Override path to binary_<pair_slug>_with_split.csv (default under training dir).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoints directory (default: under training dir, includes learning rate).",
    )
    p.add_argument("--model-name-or-path", type=str, default="macwiatrak/bacformer-large-masked-MAG")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.00015)
    p.add_argument("--max-n-proteins", type=int, default=9000)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=15)
    p.add_argument("--warmup-proportion", type=float, default=0.1)
    return p


def _resolve_paths_from_tokens(
    token1: str,
    token2: str,
    processed_base_dir: str,
    train_data_dir: str | None,
    val_data_dir: str | None,
    sheet_path: str | None,
    output_dir: str | None,
    lr: float,
) -> tuple[str, str, str, str, str, str]:
    pair_slug = sanitize_pair_name(token1, token2)
    slug1 = slugify_isolation_source_token(token1)
    slug2 = slugify_isolation_source_token(token2)
    base = Path(processed_base_dir) / f"training_{slug1}_{slug2}"
    train_dir = train_data_dir or str(base / "train")
    val_dir = val_data_dir or str(base / "validate")
    sheet = sheet_path or str(base / f"binary_{pair_slug}_with_split.csv")
    label_column = f"{pair_slug}_label"
    pt_suffix = f"_with_{pair_slug}.pt"
    out = output_dir or str(base / f"bacformer_finetuned_lr_{lr}")
    return train_dir, val_dir, sheet, out, label_column, pt_suffix


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    token1, token2 = args.isolation_sources
    train_data_dir, val_data_dir, sheet_path, output_dir, label_column, pt_suffix = (
        _resolve_paths_from_tokens(
            token1,
            token2,
            args.processed_base_dir,
            args.train_data_dir,
            args.val_data_dir,
            args.sheet_path,
            args.output_dir,
            args.lr,
        )
    )
    print("Isolation-source pair finetuning from pytorch (.pt) files")
    print(f"Tokens: {token1!r}, {token2!r} -> pair_slug={sanitize_pair_name(token1, token2)!r}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run(
        model_name_or_path=args.model_name_or_path,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=output_dir,
        sheet_path=sheet_path,
        label_column=label_column,
        pt_suffix=pt_suffix,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_n_proteins=args.max_n_proteins,
        freeze_encoder=args.freeze_encoder,
        logging_steps=args.logging_steps,
        seed=args.seed,
        n_samples=args.n_samples,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        num_workers=args.num_workers,
        warmup_proportion=args.warmup_proportion,
        max_steps=args.max_steps,
    )
