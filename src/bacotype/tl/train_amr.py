import os
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
from bacformer.modeling.config import SPECIAL_TOKENS_DICT
from bacformer.modeling.data_reader import collate_genome_samples
from bacformer.modeling.modeling_tasks import BacformerForGenomeClassification
from bacformer.modeling.trainer import BacformerTrainer
from bacformer.pp import protein_embeddings_to_inputs
from datasets import (
    load_dataset,  # Hugging Face datasets library
)
from tap import Tap
from transformers import AutoConfig, EarlyStoppingCallback, EvalPrediction, TrainingArguments

############################################################## Transform sample -unchanged from original script ##############################################################


def transform_sample(
    max_n_proteins: int = 9000,  # For Klebsiella pneumoniae, we use 9000 proteins
    max_n_contigs: int = 1000,
    row: dict = None,
):
    """
    Transform the sample to Bacformer inputs.

    If the row contains pre-processed bacformer inputs (contig_ids/attention_mask or
    special_tokens_mask/token_type_ids from prepare_klebsiella_ast_splits), they are
    used directly. Otherwise calls protein_embeddings_to_inputs (which expects raw
    per-protein embeddings; parquet must be built from .pt files that store these).
    """
    # Use pre-processed inputs when present (parquet built with full .pt contents).
    # This avoids passing the processed sequence back into protein_embeddings_to_inputs,
    # which expects raw per-protein embeddings and can raise IndexError otherwise.
    if "contig_ids" in row or "attention_mask" in row:
        # Large model: protein_embeddings is list of arrays -> stack to tensor
        prot = row["protein_embeddings"]
        if isinstance(prot, list):
            protein_embeddings = torch.tensor(
                np.stack([np.asarray(p, dtype=np.float32) for p in prot]),
                dtype=torch.float32,
            )
        else:
            protein_embeddings = torch.as_tensor(prot, dtype=torch.float32)
        result = {"protein_embeddings": protein_embeddings}
        if "contig_ids" in row:
            result["contig_ids"] = torch.as_tensor(row["contig_ids"], dtype=torch.long)
        if "attention_mask" in row:
            result["attention_mask"] = torch.as_tensor(row["attention_mask"], dtype=torch.float32)
        return result
    if "special_tokens_mask" in row and "token_type_ids" in row:
        prot = row["protein_embeddings"]
        if isinstance(prot, list):
            protein_embeddings = torch.tensor(
                np.stack([np.asarray(p, dtype=np.float32) for p in prot]),
                dtype=torch.float32,
            )
        else:
            protein_embeddings = torch.as_tensor(prot, dtype=torch.float32)
        result = {
            "protein_embeddings": protein_embeddings,
            "special_tokens_mask": torch.as_tensor(row["special_tokens_mask"], dtype=torch.long),
            "token_type_ids": torch.as_tensor(row["token_type_ids"], dtype=torch.long),
        }
        if "attention_mask" in row:
            result["attention_mask"] = torch.as_tensor(row["attention_mask"], dtype=torch.float32)
        return result

    inputs = protein_embeddings_to_inputs(
        protein_embeddings=row["protein_embeddings"],
        max_n_proteins=max_n_proteins,
        max_n_contigs=max_n_contigs,
        bacformer_model_type="large",
    )
    result = {}
    for k, v in inputs.items():
        if k == "protein_embeddings":
            result[k] = v.squeeze().to(torch.float32).detach()
        elif k in ["special_tokens_mask", "token_type_ids"]:
            result[k] = v.squeeze().to(torch.long).detach()
        elif k == "attention_mask":
            result[k] = v.squeeze().to(torch.float32).detach()
        else:
            if isinstance(v, torch.Tensor):
                result[k] = v.squeeze().detach()
            else:
                result[k] = v.squeeze()
    return result


############################################################## Compute metrics - unchanged from original script ##############################################################


def compute_metrics_binary_genome_pred(preds: EvalPrediction, ignore_index: int = -100, prefix: str = "eval"):
    """Compute metrics for a single-logit binary classifier."""
    with torch.no_grad():
        logits = torch.tensor(preds.predictions).flatten()  # (N,)
        labels = torch.tensor(preds.label_ids).flatten().long()
        # mask out ignore_index if any (HF usually won't have it, but safe)
        if (labels == ignore_index).any():
            keep = labels != ignore_index
            logits, labels = logits[keep], labels[keep]
        prob = torch.sigmoid(logits)  # P(y=1)
        pred = (prob >= 0.5).long()
        # compute by hand (avoids dtype/shape pitfalls)
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


############################################################## Main (run) function ##############################################################
def run(
    model_name_or_path: str,
    train_data_dir: str,
    val_data_dir: str,
    output_dir: str,
    ast_sheet_path: str,
    lr: float = 0.00015,
    batch_size: int = 1,
    grad_accumulation_steps: int = 8,
    drug: str = "ampicillin",
    max_n_proteins: int = 6000,
    freeze_encoder: bool = False,
    logging_steps: int = 10,
    n_samples: int = 10000,
    seed: int = 1,
    early_stopping_patience: int = 10,  # Add this parameter
    eval_steps: int = 150,  # Evaluate every 50 steps, only used if use_epochs is False (full set mode)
    num_workers: int = 16,  # Number of workers for data loading
    warmup_proportion: float = 0.1,  # Proportion of steps to use for learning rate warmup
    max_steps: int = 20000,
):
    """Example script to fine-tune Bacformer on AMR data."""
    print(f"Loading model from: {model_name_or_path} - training will start from this model")
    print(f"Predicting AMR for drug: {drug}")
    print(f"Training set: {train_data_dir}")
    print(f"Validation set: {val_data_dir}")
    print(f"Finetuning model, n_samples = {n_samples}")
    print(f"Freezing encoder: {freeze_encoder}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Number of workers: {num_workers}")
    print(f"Eval steps: {eval_steps}")
    print(f"Grad accumulation steps: {grad_accumulation_steps}")
    print(f"Batch size: {batch_size}")
    # max steps
    print(f"Max steps: {max_steps}")
    print(f"Warmup proportion: {warmup_proportion} / {max_steps} (of max steps)")
    print(f"Interim and final models will be saved in the output directory: {output_dir}")
    print(f"AST sheet (with splits): {ast_sheet_path}")
    print("------------------------------------------------\n")
    ############################################################## Set up the training and validation datasets ##############################################################

    # Load AST sheet with splits and log per-split counts for this drug
    if not ast_sheet_path:
        raise ValueError("ast_sheet_path must be provided and point to binary_ast_with_split.csv")
    if not os.path.exists(ast_sheet_path):
        raise FileNotFoundError(f"AST sheet not found at {ast_sheet_path}")

    ast_df = pd.read_csv(ast_sheet_path)
    if "train_val_eval" not in ast_df.columns:
        raise ValueError("AST sheet must contain 'train_val_eval' column. Run prepare_klebsiella_ast_splits.py first.")
    if "Sample" not in ast_df.columns:
        if "phenotype-BioSample_ID" in ast_df.columns:
            ast_df["Sample"] = ast_df["phenotype-BioSample_ID"].astype(str)
        else:
            raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID' column.")
    if drug not in ast_df.columns:
        raise ValueError(f"Drug column '{drug}' not found in AST sheet.")

    counts = ast_df.loc[ast_df[drug].notna()].groupby("train_val_eval")["Sample"].nunique().to_dict()
    train_count = counts.get("train", 0)
    val_count = counts.get("validate", 0)
    eval_count = counts.get("evaluate", 0)
    print(
        f"Samples with non-missing '{drug}' labels - "
        f"train: {train_count}, validate: {val_count}, evaluate: {eval_count}"
    )

    # Build file lists from AST sheet (replaces glob: only files for samples that have this drug).
    # Data is always loaded lazily via streaming; we never load the full dataset into memory.
    def build_file_list(split_name: str):
        ids = ast_df[(ast_df["train_val_eval"] == split_name) & ast_df[drug].notna()]["Sample"].astype(str).unique()
        if split_name == "train":
            base_dir = train_data_dir
        elif split_name == "validate":
            base_dir = val_data_dir
        else:
            base_dir = None
        if base_dir is None:
            return []
        files = []
        for sid in ids:
            path = os.path.join(base_dir, f"{sid}.parquet")
            if os.path.exists(path):
                files.append(path)
            else:
                print(f"WARNING: Expected parquet file missing for Sample {sid}: {path}")
        return files

    # Check directories exist
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Train data directory {train_data_dir} does not exist")
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(f"Validation data directory {val_data_dir} does not exist")

    # Dummy mode: first 10 train samples with this drug; same 10 for train and val.
    # Full mode: all train/validate samples with this drug. Both use same lazy streaming path.
    if n_samples == 10:
        print("Using dummy test mode with 10 samples (same set for train and val).")
        train_ids = (
            ast_df[(ast_df["train_val_eval"] == "train") & ast_df[drug].notna()]["Sample"].astype(str).unique()[:10]
        )
        train_files = [
            os.path.join(train_data_dir, f"{sid}.parquet")
            for sid in train_ids
            if os.path.exists(os.path.join(train_data_dir, f"{sid}.parquet"))
        ]
        val_files = train_files
        eval_strategy = "epoch"
        use_epochs = True
        num_train_epochs = 100
    else:
        print("Full set mode: Using lazy dataset with streaming")
        train_files = build_file_list("train")
        val_files = build_file_list("validate")
        eval_strategy = "steps"
        use_epochs = False

    if not train_files:
        raise RuntimeError("No parquet files found for training (check AST sheet and drug column).")

    print(f"Number of train files (samples with '{drug}'): {len(train_files)}")
    print(f"Number of validation files (samples with '{drug}'): {len(val_files)}")

    # Lazy streaming: load_dataset streams from the file list; no full dataset load.
    # Load all columns so pre-processed keys (contig_ids, attention_mask, etc.) are
    # available when parquet was built with prepare_klebsiella_ast_splits saving full .pt contents.
    train_dataset = load_dataset(
        "parquet",
        data_files=train_files,
        split="train",
        streaming=True,
    )
    val_dataset = load_dataset(
        "parquet",
        data_files=val_files,
        split="train",
        streaming=True,
    )
    print("Streaming dataset ready; relying on DataLoader workers for parallelism")

    def filter_missing(example):
        return example[drug] is not None

    def rename_drug_to_label(example):
        example["label"] = example[drug]
        return {k if k != drug else "label": v for k, v in example.items()}

    train_dataset = train_dataset.filter(filter_missing).map(rename_drug_to_label)
    val_dataset = val_dataset.filter(filter_missing).map(rename_drug_to_label)
    transform_fn = partial(transform_sample, max_n_proteins, 1000)
    train_dataset = train_dataset.map(transform_fn, batched=False, with_indices=False)
    val_dataset = val_dataset.map(transform_fn, batched=False, with_indices=False)

    if not use_epochs:
        # Take a sample to check the structure
        try:
            sample = next(iter(train_dataset.take(1)))
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Drug column '{drug}' exists: {drug in sample}")
            if drug in sample:
                print(f"Sample drug value: {sample[drug]}")
            print(
                f"Protein embeddings shape: "
                f"{sample['protein_embeddings'].shape if 'protein_embeddings' in sample else 'Not found'}"
            )
        except StopIteration:
            print("Train dataset is empty")

    ############################################################## Configure the Bacformer model for training##############################################################
    # load the Bacformer model
    # for this task we use the Bacformer model trained on masked complete genomes
    # with a genome classification head
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.num_labels = 1
    config.problem_type = "binary_classification"

    bacformer_model = BacformerForGenomeClassification.from_pretrained(
        model_name_or_path, config=config, trust_remote_code=True
    ).to(torch.bfloat16)

    # freeze the Bacformer model parameters
    if freeze_encoder:
        for param in bacformer_model.bacformer.parameters():
            param.requires_grad = False

    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable parameters:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    # define the output directory for the model and metrics
    os.makedirs(output_dir, exist_ok=True)

    # create a trainer
    # get training args into a dictionary to pass. Using dictionary so we can flexibly decide on epoch / steps approach
    training_args_dict = {
        "output_dir": output_dir,
        "eval_strategy": eval_strategy,
        "save_strategy": eval_strategy,
        "save_total_limit": 1,
        "learning_rate": lr,  # This will be scaled by setup_multi_gpu_training if using multiple GPUs
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
        "bf16": True if torch.cuda.is_available() else False,
        "metric_for_best_model": "eval_auroc",
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "logging_nan_inf_filter": False,  # Show NaN values in logs
        "report_to": ["tensorboard"],  # Only use tensorboard for reporting
    }
    # Use either epochs or max_steps depending on dataset size (dummy set or full set / steaming or deterministic)
    if use_epochs:
        training_args_dict["num_train_epochs"] = num_train_epochs
        print(f"Using num_train_epochs: {num_train_epochs}")
    else:
        training_args_dict["max_steps"] = max_steps
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["save_steps"] = eval_steps
        # Add warmup configuration
        warmup_steps = int(max_steps * warmup_proportion)
        training_args_dict["warmup_steps"] = warmup_steps
        training_args_dict["lr_scheduler_type"] = "linear"  # Use linear warmup
        print(f"Warmup steps: {warmup_steps} (of {max_steps} total steps)")
        print(f"Using max_steps: {max_steps}, eval_steps: {eval_steps}, warmup_steps: {warmup_steps}")
    # Now use the training args dict to create the training arguments
    training_args = TrainingArguments(**training_args_dict)

    # define a collate function for the dataset
    collate_genome_samples_fn = partial(collate_genome_samples, SPECIAL_TOKENS_DICT["PAD"], 1000)
    # datasets for trainer are defined above depending on dummy set or full set / steaming or deterministic
    trainer = BacformerTrainer(
        model=bacformer_model,
        data_collator=collate_genome_samples_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics_binary_genome_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()


############################################################## Argument parser ##############################################################
class ArgumentParser(Tap):
    """Argument parser for training Bacformer."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    model_name_or_path: str = "macwiatrak/bacformer-large-masked-MAG"
    train_data_dir: str = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/ast_training/train"
    val_data_dir: str = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/ast_training/validate"
    output_dir: str = "/tmp/train-output/"
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    lr: float = 0.00015  # default for frozen embeddings, reasonable range is 0.001-0.1
    # lr: float = 0.00015 # default for finetuned embeddings, reasonable range is 0.0001-0.01
    drug: str = "ceftriaxone"
    max_n_proteins: int = 9000
    freeze_encoder: bool = False
    logging_steps: int = 10
    n_samples: int = 10000
    ast_sheet_path: str = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/binary_ast_with_split.csv"
    seed: int = 1
    max_steps: int = 100000  # 20,000 x 8 samples processed (grad accumulation steps) = 160,000 samples processed, 160,000 / 5000 samples = 32 epochs
    early_stopping_patience: int = 30  # Add this parameter with default value
    eval_steps: int = 250  # Evaluate every 50 steps, only used if use_epochs is False (full set mode)
    num_workers: int = 15  # Number of workers for data loading
    warmup_proportion: float = 0.1  # Proportion of steps to use for learning rate warmup


############################################################## Main function ##############################################################
if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    print("Running finetuning with partited dataset")
    # print the timestamp
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run(
        model_name_or_path=args.model_name_or_path,
        train_data_dir=args.train_data_dir,  # train data directory
        val_data_dir=args.val_data_dir,  # validation data directory
        output_dir=args.output_dir,  # output directory
        ast_sheet_path=args.ast_sheet_path,
        lr=args.lr,  # learning rate
        batch_size=args.batch_size,  # batch size
        grad_accumulation_steps=args.grad_accumulation_steps,  # gradient accumulation steps
        drug=args.drug,  # drug
        max_n_proteins=args.max_n_proteins,  # max number of proteins - 6000 by default
        freeze_encoder=args.freeze_encoder,  # false by default (fine-tuning encoder)
        logging_steps=args.logging_steps,
        seed=args.seed,
        n_samples=args.n_samples,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        num_workers=args.num_workers,
        warmup_proportion=args.warmup_proportion,
        max_steps=args.max_steps,
    )
