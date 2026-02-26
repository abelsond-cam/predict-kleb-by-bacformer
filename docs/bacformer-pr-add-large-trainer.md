# Bacformer PR: Add BacformerLargeTrainer

## PR Title

**Add BacformerLargeTrainer for Large model (contig_ids, no special_tokens_mask)**

---

## PR Description

### Summary

Adds `BacformerLargeTrainer` to support training `BacformerLargeForGenomeClassification`. The existing `BacformerTrainer` is built for the base 26M model and expects `special_tokens_mask` and `token_type_ids`, which the Large model does not use. The Large model instead expects `contig_ids` and `attention_mask`, causing `KeyError: 'special_tokens_mask'` when using `BacformerTrainer` with Large.

### Motivation

- `BacformerLargeForGenomeClassification` has a different forward signature: `contig_ids`, `attention_mask` (optional), no `special_tokens_mask` or `token_type_ids`.
- `protein_embeddings_to_inputs` with `bacformer_model_type="large"` returns `protein_embeddings`, `contig_ids`, and `attention_mask` — not the base-model fields.
- Users fine-tuning the Large model for genome classification (e.g. AMR prediction) need a trainer compatible with this interface.

### Changes

- New `BacformerLargeTrainer` class that:
  - Passes `protein_embeddings`, `labels`, `attention_mask`, `contig_ids` to the model.
  - Accepts `token_type_ids` as fallback for `contig_ids` for backwards compatibility.
  - Handles both tuple and `ModelOutput` return formats (`return_dict` True/False).

### Testing

Tested with `macwiatrak/bacformer-large-masked-MAG` fine-tuning for binary classification on .pt files containing `protein_embeddings`, `attention_mask`, and `contig_ids` (from `protein_embeddings_to_inputs` with `bacformer_model_type="large"`).

---

## Diff for `bacformer/modeling/trainer.py`

```diff
--- a/bacformer/modeling/trainer.py
+++ b/bacformer/modeling/trainer.py
@@ -1,8 +1,10 @@
 from torch.utils.data import Dataset
 from transformers import DataCollator, Trainer, TrainingArguments, is_datasets_available

 from bacformer.modeling.modeling_base import BacformerModel
+from bacformer.modeling.modeling_large import BacformerLargeForGenomeClassification
 from bacformer.modeling.modeling_pretraining import BacformerForCausalProteinFamilyModeling

 if is_datasets_available():
@@ -92,3 +94,57 @@ class BacformerCausalProteinFamilyTrainer(Trainer):
         if return_outputs:
             return outputs[0], outputs[1:]
         return outputs[0]
+
+
+class BacformerLargeTrainer(Trainer):
+    """Trainer for BacformerLargeForGenomeClassification.
+
+    Bacformer Large uses contig_ids and attention_mask (no special_tokens_mask/token_type_ids).
+    The default BacformerTrainer is for the base 26M model which expects different inputs.
+    """
+
+    def __init__(
+        self,
+        model: BacformerLargeForGenomeClassification,
+        args: TrainingArguments = None,
+        data_collator: DataCollator | None = None,
+        train_dataset: Dataset | None = None,
+        eval_dataset: Dataset | dict[str, Dataset] | None = None,
+        **kwargs,
+    ):
+        super().__init__(
+            model=model,
+            args=args,
+            data_collator=data_collator,
+            train_dataset=train_dataset,
+            eval_dataset=eval_dataset,
+            **kwargs,
+        )
+
+    def compute_loss(
+        self,
+        model: BacformerLargeForGenomeClassification,
+        inputs: dict,
+        num_items_in_batch: int = None,
+        return_outputs: bool = False,
+    ):
+        """Compute loss for Bacformer Large genome classification."""
+        protein_embeddings = inputs.pop("protein_embeddings")
+        labels = inputs.pop("labels")
+        attention_mask = inputs.pop("attention_mask", None)
+        contig_ids = inputs.pop("contig_ids", inputs.pop("token_type_ids", None))
+
+        outputs = model(
+            protein_embeddings=protein_embeddings,
+            labels=labels,
+            attention_mask=attention_mask,
+            contig_ids=contig_ids,
+        )
+
+        # Handle both tuple (return_dict=False) and ModelOutput (return_dict=True)
+        if isinstance(outputs, tuple):
+            loss = outputs[0]
+        elif isinstance(outputs, dict):
+            loss = outputs["loss"]
+        else:
+            loss = outputs.loss
+
+        if return_outputs:
+            return loss, outputs
+        return loss
```

---

## Steps to apply and open the PR

1. **Fork** https://github.com/macwiatrak/Bacformer
2. **Clone** your fork and create a branch:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Bacformer.git
   cd Bacformer
   git checkout -b add-bacformer-large-trainer
   ```
3. **Edit** `bacformer/modeling/trainer.py` and add the import plus the new class (as in the diff above).
4. **Commit and push**:
   ```bash
   git add bacformer/modeling/trainer.py
   git commit -m "Add BacformerLargeTrainer for Large model (contig_ids, no special_tokens_mask)"
   git push -u origin add-bacformer-large-trainer
   ```
5. **Open a PR** on https://github.com/macwiatrak/Bacformer from your branch, and paste the PR description above.
