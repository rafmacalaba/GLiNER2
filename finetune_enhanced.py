#!/usr/bin/env python3
"""
Fine-tune GLiNER2 for Dataset Mention Extraction
Enhanced version with better logging and configuration

This script fine-tunes GLiNER2 on dataset mention extraction data using JSON structures format.
Combines the best practices from the unofficial GLiNER2 fine-tuning approach.

Usage:
    python finetune_enhanced.py --base-model fastino/gliner2-base-v1 --epochs 5
    
    # With custom paths
    python finetune_enhanced.py \
        --base-model fastino/gliner2-base-v1 \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --output-dir models/my-model \
        --epochs 5
"""

from datetime import datetime
from pathlib import Path
import os
import typer
from loguru import logger
from transformers import TrainingArguments
from gliner2 import GLiNER2
from gliner2.trainer import ExtractorDataset, ExtractorDataCollator, ExtractorTrainer

# Fix for multi-GPU DataParallel issue with GLiNER2
# Force single GPU to avoid 'DataParallel' object has no attribute 'processor' error
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    # Model configuration
    base_model: str = typer.Option(
        "fastino/gliner2-base-v1",
        "--base-model",
        "-m",
        help="Pre-trained model to fine-tune"
    ),
    
    # Data paths
    train_data: str = typer.Option(
        "data/train.jsonl",
        "--train-data",
        help="Path to training JSONL file"
    ),
    val_data: str = typer.Option(
        "data/val.jsonl",
        "--val-data",
        help="Path to validation JSONL file"
    ),
    
    # Output configuration
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (auto-generated if not specified)"
    ),
    
    # Training hyperparameters
    epochs: int = typer.Option(
        5,
        "--epochs",
        "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        "-b",
        help="Per-device batch size"
    ),
    gradient_accumulation: int = typer.Option(
        4,
        "--gradient-accumulation",
        "-g",
        help="Gradient accumulation steps"
    ),
    
    # Learning rates
    encoder_lr: float = typer.Option(
        1e-5,
        "--encoder-lr",
        help="Learning rate for encoder"
    ),
    custom_lr: float = typer.Option(
        5e-5,
        "--custom-lr",
        help="Learning rate for task layers"
    ),
    
    # Training strategy
    freeze_encoder: bool = typer.Option(
        False,
        "--freeze-encoder",
        help="Freeze encoder (classifier-only fine-tuning)"
    ),
    
    # Other options
    warmup_steps: int = typer.Option(
        500,
        "--warmup-steps",
        help="Number of warmup steps"
    ),
    eval_steps: int = typer.Option(
        500,
        "--eval-steps",
        help="Evaluation interval"
    ),
    save_steps: int = typer.Option(
        500,
        "--save-steps",
        help="Checkpoint save interval"
    ),
    fp16: bool = typer.Option(
        False,
        "--fp16",
        help="Use mixed precision training (disable if you get FP16 scaler errors)"
    ),
    report_to: str = typer.Option(
        "none",
        "--report-to",
        help="Reporting tool (none/wandb/tensorboard)"
    ),
):
    """
    Fine-tune GLiNER2 for dataset mention extraction.
    
    Your training data should be in JSONL format with json_structures:
    {"input": "text", "output": {"json_structures": [...]}}
    """
    
    logger.info("=" * 60)
    logger.info("GLiNER2 Fine-Tuning for Dataset Mention Extraction")
    logger.info("=" * 60)
    
    # Generate run name and output directory
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Resolve paths
    train_path = Path(train_data).resolve()
    val_path = Path(val_data).resolve() if val_data else None
    
    # Auto-generate output directory if not specified
    if output_dir is None:
        output_dir = f"models/dataset-mention-{datestamp}"
    output_path = Path(output_dir).resolve()
    
    # Create run name
    run_name = f"gliner2-dataset-{datestamp}-ep{epochs}"
    
    logger.info("\n📋 Configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Train data: {train_path}")
    logger.info(f"  Val data: {val_path}")
    logger.info(f"  Output dir: {output_path}")
    logger.info(f"  Run name: {run_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation}")
    logger.info(f"  Encoder LR: {encoder_lr}")
    logger.info(f"  Custom LR: {custom_lr}")
    logger.info(f"  Freeze encoder: {freeze_encoder}")
    logger.info(f"  FP16: {fp16}")
    
    # 1. Load datasets
    logger.info("\n📊 Loading datasets...")
    
    if not train_path.exists():
        logger.error(f"❌ Training data not found: {train_path}")
        raise typer.Exit(1)
    
    train_dataset = ExtractorDataset(str(train_path))
    logger.info(f"  ✅ Training examples: {len(train_dataset)}")
    
    eval_dataset = None
    if val_path and val_path.exists():
        eval_dataset = ExtractorDataset(str(val_path))
        logger.info(f"  ✅ Validation examples: {len(eval_dataset)}")
    else:
        logger.warning(f"  ⚠️  No validation data found")
    
    # 2. Load model
    logger.info(f"\n📥 Loading model: {base_model}")
    model = GLiNER2.from_pretrained(base_model)
    logger.info("  ✅ Model loaded successfully")
    
    # 3. Configure training
    logger.info("\n⚙️  Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=custom_lr,  # This gets overridden by trainer
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=fp16,
        report_to=report_to,
        run_name=run_name,
    )
    
    # 4. Create trainer
    logger.info("\n🏋️  Creating trainer...")
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ExtractorDataCollator(),
        encoder_lr=encoder_lr,
        custom_lr=custom_lr,
        weight_decay=0.01,
        finetune_classifier=freeze_encoder,
    )
    
    # 5. Train
    logger.info("\n🚀 Starting training...")
    logger.info("=" * 60)
    
    trainer.train()
    
    # 6. Save final model
    final_model_path = output_path / "final"
    logger.info(f"\n💾 Saving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))
    trainer.save_state()
    
    # 7. Evaluate
    if eval_dataset:
        logger.info("\n📈 Final evaluation...")
        eval_metrics = trainer.evaluate()
        logger.info(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
        logger.info(f"  All metrics: {eval_metrics}")
    
    # 8. Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info("=" * 60)
    logger.info(f"\n📁 Model saved to: {final_model_path}")
    logger.info("\n🔧 To use the fine-tuned model:")
    logger.info(f"  from gliner2 import GLiNER2")
    logger.info(f"  extractor = GLiNER2.from_pretrained('{final_model_path}')")
    logger.info(f"  results = extractor.extract(text, schema)")
    logger.info("\nDone! 🎉")


if __name__ == "__main__":
    app()
