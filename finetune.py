#!/usr/bin/env python3
"""
Fine-tune GLiNER2 - Configurable version of train.py

Based on the working train.py but with configurable paths and hyperparameters.

Usage:
    python finetune.py fastino/gliner2-base-v1 \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --epochs 5
"""

from datetime import datetime
import typer
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os

from transformers.training_args import TrainingArguments
from gliner2 import GLiNER2
from gliner2.trainer import ExtractorDataset, ExtractorDataCollator, ExtractorTrainer

load_dotenv()
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    base_model_id: str = typer.Argument(..., help="Model ID (e.g., fastino/gliner2-base-v1)"),
    
    # Data paths - now configurable!
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
    
    # Output
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        help="Output directory (auto-generated if not specified)"
    ),
    
    # Training hyperparameters - now configurable!
    epochs: int = typer.Option(
        3,
        "--epochs",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Per-device batch size"
    ),
    gradient_accumulation: int = typer.Option(
        1,
        "--gradient-accumulation",
        help="Gradient accumulation steps"
    ),
    
    # Learning rates - now configurable!
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
        help="Save checkpoint interval"
    ),
    freeze_encoder: bool = typer.Option(
        False,
        "--freeze-encoder",
        help="Freeze encoder (classifier-only fine-tuning)"
    ),
    report_to: str = typer.Option(
        "none",
        "--report-to",
        help="Reporting tool (none/wandb/tensorboard)"
    ),
):
    """
    Fine-tune GLiNER2 on your dataset.
    
    Example:
        python finetune.py fastino/gliner2-base-v1 --epochs 5
        
        python finetune.py fastino/gliner2-large-v1 \\
            --train-data /kaggle/input/my-data/train.jsonl \\
            --val-data /kaggle/input/my-data/val.jsonl \\
            --output-dir /kaggle/working/my-model \\
            --epochs 5 \\
            --batch-size 4 \\
            --gradient-accumulation 4
    """
    
    print("=" * 60)
    print("GLiNER2 Fine-Tuning")
    print("=" * 60)

    datestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    logger.info("Configuration:")
    logger.info(f"  Model: {base_model_id}")
    logger.info(f"  Train data: {train_data}")
    logger.info(f"  Val data: {val_data}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation}")
    logger.info(f"  Encoder LR: {encoder_lr}")
    logger.info(f"  Custom LR: {custom_lr}")
    logger.info(f"  Freeze encoder: {freeze_encoder}")

    # 1. Datasets
    logger.info("\nLoading datasets...")
    train_files = [train_data]
    eval_files = [val_data]
    
    train_dataset = ExtractorDataset(train_files)
    eval_dataset = ExtractorDataset(eval_files)
    logger.info(f"  Training examples: {len(train_dataset)}")
    logger.info(f"  Validation examples: {len(eval_dataset)}")

    # 2. Model
    logger.info(f"\nLoading model: {base_model_id}")
    model = GLiNER2.from_pretrained(base_model_id)
    logger.info("  Model loaded successfully")

    # 3. Data collator
    data_collator = ExtractorDataCollator()

    # 4. Output directory
    if output_dir is None:
        run_name = f"gliner2-{datestamp}-tr{len(train_dataset)}-ev{len(eval_dataset)}-ep{epochs}"
        output_dir = f"checkpoints/{run_name}"
    else:
        run_name = Path(output_dir).name
    
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Run name: {run_name}")

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=custom_lr,  # global LR (group-specific LRs override)
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        # fp16=True,  # Enable if you have GPU with fp16 support
        report_to=report_to,
        run_name=run_name,
    )

    # 6. Trainer
    logger.info("\nCreating trainer...")
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        encoder_lr=encoder_lr,
        custom_lr=custom_lr,
        weight_decay=0.01,
        finetune_classifier=freeze_encoder,
    )

    # 7. Train
    logger.info("\nStarting training...")
    logger.info("=" * 60)
    
    trainer.train()
    
    # 8. Save
    final_path = Path(output_dir) / "final"
    logger.info(f"\nSaving model to: {final_path}")
    trainer.save_model(str(final_path))
    trainer.save_state()
    logger.info("Model saved successfully")

    # 9. Evaluate
    logger.info("\nFinal evaluation...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"\nModel saved to: {final_path}")
    logger.info("\nTo use the model:")
    logger.info(f"  from gliner2 import GLiNER2")
    logger.info(f"  model = GLiNER2.from_pretrained('{final_path}')")


if __name__ == "__main__":
    app()
