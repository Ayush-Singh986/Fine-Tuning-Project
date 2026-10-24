import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from finetune.data import load_preference_rows, load_sft_rows


def load_base(model_name: str, *, quantized: bool):
    if quantized and not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; no CUDA device was detected.")
    quantization = None
    if quantized:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA SFT adapter or DPO-tune an SFT adapter.")
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft-adapter", type=Path, help="Required for DPO; output from the SFT stage.")
    parser.add_argument("--data", type=Path, required=True, help="SFT or preference JSONL.")
    parser.add_argument("--validation-data", type=Path, help="Optional held-out validation JSONL for this training stage.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true", help="Load full precision weights; needs substantially more GPU memory.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    use_4bit = not args.no_4bit
    if args.stage == "sft":
        model, tokenizer = load_base(args.model, quantized=use_4bit)
        rows = load_sft_rows(args.data, tokenizer)
        validation_rows = (
            load_sft_rows(args.validation_data, tokenizer)
            if args.validation_data
            else None
        )
        model = get_peft_model(model, lora_config())
        config = SFTConfig(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if validation_rows else "no",
            gradient_checkpointing=True,
            max_seq_length=args.max_length,
            dataset_text_field="text",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=Dataset.from_list(rows),
            eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
            processing_class=tokenizer,
        )
    else:
        if args.sft_adapter is None:
            parser.error("--sft-adapter is required for the DPO stage.")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        rows = load_preference_rows(args.data, tokenizer)
        validation_rows = (
            load_preference_rows(args.validation_data, tokenizer)
            if args.validation_data
            else None
        )
        base, _ = load_base(args.model, quantized=use_4bit)
        model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
        config = DPOConfig(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate / 10,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if validation_rows else "no",
            gradient_checkpointing=True,
            max_length=args.max_length,
            max_prompt_length=args.max_length // 2,
            beta=0.1,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=Dataset.from_list(rows),
            eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    print(f"Saved {args.stage.upper()} adapter to {args.output / 'adapter'}")


if __name__ == "__main__":
    main()
