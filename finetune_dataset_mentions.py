#!/usr/bin/env python3
"""
Fine-tune GLiNER2 for Dataset Mention Extraction

This script fine-tunes GLiNER2 on the generated dataset mention extraction data
using JSON structures format.

Usage:
    python finetune_dataset_mentions.py --train-data data/train.jsonl --val-data data/val.jsonl
"""

import argparse
from gliner2 import Extractor, ExtractorConfig
from gliner2.trainer import ExtractorTrainer, ExtractorDataset, ExtractorDataCollator
from transformers import TrainingArguments


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER2 for dataset mention extraction")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation JSONL file (optional)")
    parser.add_argument("--model-name", type=str, default="fastino/gliner2-base-v1", 
                        help="Pre-trained model to fine-tune (default: fastino/gliner2-base-v1)")
    parser.add_argument("--output-dir", type=str, default="./models/dataset-mention-extractor",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, 
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, 
                        help="Learning rate for encoder (default: 1e-5)")
    parser.add_argument("--custom-lr", type=float, default=5e-5,
                        help="Learning rate for task layers (default: 5e-5)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps (default: 500)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps (default: 500)")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint steps (default: 500)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--freeze-encoder", action="store_true", 
                        help="Freeze encoder and only train task layers (classifier-only fine-tuning)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLiNER2 Fine-Tuning for Dataset Mention Extraction")
    print("=" * 60)
    
    # 1. Load pre-trained model
    print(f"\n📥 Loading pre-trained model: {args.model_name}")
    model = Extractor.from_pretrained(args.model_name)
    print("✅ Model loaded successfully")
    
    # 2. Prepare datasets
    print(f"\n📊 Loading training data: {args.train_data}")
    train_dataset = ExtractorDataset(args.train_data)
    
    eval_dataset = None
    if args.val_data:
        print(f"📊 Loading validation data: {args.val_data}")
        eval_dataset = ExtractorDataset(args.val_data)
    
    # 3. Configure training
    print(f"\n⚙️  Configuring training:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  - Encoder LR: {args.encoder_lr}")
    print(f"  - Custom LR: {args.custom_lr}")
    print(f"  - Freeze encoder: {args.freeze_encoder}")
    print(f"  - FP16: {args.fp16}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_steps=args.save_steps,
        logging_steps=100,
        learning_rate=args.custom_lr,  # This gets overridden by trainer
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # 4. Create trainer
    print(f"\n🏋️  Creating trainer...")
    trainer = ExtractorTrainer(
        model=model,
        encoder_lr=args.encoder_lr,
        custom_lr=args.custom_lr,
        weight_decay=0.01,
        finetune_classifier=args.freeze_encoder,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ExtractorDataCollator(),
    )
    
    # 5. Train
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    trainer.train()
    
    # 6. Save final model
    print(f"\n💾 Saving final model to: {args.output_dir}/final")
    model.save_pretrained(f"{args.output_dir}/final")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {args.output_dir}/final")
    print("\nTo use the fine-tuned model:")
    print(f"  from gliner2 import GLiNER2")
    print(f"  extractor = GLiNER2.from_pretrained('{args.output_dir}/final')")
    print(f"  results = extractor.extract_json(text, schema)")


if __name__ == "__main__":
    main()
