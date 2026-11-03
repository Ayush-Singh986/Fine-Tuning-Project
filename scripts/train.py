# Import argparse for parsing command-line arguments
import argparse
# Import Path for cross-platform file path handling
from pathlib import Path

# Import PyTorch for tensor operations and GPU detection
import torch
# Import Dataset for creating training datasets from lists
from datasets import Dataset
# Import PEFT (Parameter-Efficient Fine-Tuning) components for LoRA
# LoraConfig: Configuration for LoRA adapters
# PeftModel: Model wrapper for loading LoRA adapters
# get_peft_model: Wraps a model with LoRA adapters
# prepare_model_for_kbit_training: Prepares quantized models for training
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# Import Transformers components for model loading and quantization
# AutoModelForCausalLM: Generic causal language model loader
# AutoTokenizer: Generic tokenizer loader
# BitsAndBytesConfig: Configuration for 4-bit/8-bit quantization
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# Import TRL (Transformer Reinforcement Learning) components
# DPOConfig/DPOTrainer: For Direct Preference Optimization
# SFTConfig/SFTTrainer: For Supervised Fine-Tuning
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

# Import our data loading functions
from finetune.data import load_preference_rows, load_sft_rows


def load_base(model_name: str, *, quantized: bool):
    """Load the base model and tokenizer with optional quantization.
    
    This function handles the complexity of loading models for training:
    - Detects available hardware (GPU/CPU)
    - Chooses optimal data types (bf16, fp16, fp32)
    - Configures 4-bit quantization for memory-efficient training
    - Prepares the model for training
    
    QLoRA (Quantized LoRA) allows training large models on smaller GPUs
    by loading the model in 4-bit precision while training LoRA adapters
    in higher precision.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-3B-Instruct")
        quantized: Whether to use 4-bit quantization (QLoRA)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If quantized is True but no GPU is available
    """
    # QLoRA requires a GPU - it cannot run on CPU
    # This is because 4-bit matrix operations are GPU-only
    if quantized and not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; no CUDA device was detected.")
    
    # Initialize quantization config (will be None if not quantizing)
    quantization = None
    if quantized:
        # Choose the compute data type based on GPU support
        # bfloat16 is preferred on modern GPUs (Ampere+) for numerical stability
        # float16 is used on older GPUs
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Configure 4-bit quantization using bitsandbytes
        quantization = BitsAndBytesConfig(
            # Load weights in 4-bit precision to save memory
            load_in_4bit=True,
            # Use NormalFloat4 quantization (optimal for normally distributed weights)
            bnb_4bit_quant_type="nf4",
            # Use double quantization for additional memory savings
            bnb_4bit_use_double_quant=True,
            # Use the optimal compute dtype for matrix operations
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        # For non-quantized training, choose the best data type
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set the pad token if not already set
    # Many models don't have a pad token by default
    # Using EOS token as pad is a common convention
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        # Automatically place the model on available devices
        # This handles multi-GPU and CPU-offloading
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    
    # Disable KV cache during training to save memory
    # KV cache is only useful for inference, not training
    model.config.use_cache = False
    
    # For quantized models, prepare them for training
    # This enables gradient computation for the non-quantized parts
    if quantized:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def lora_config() -> LoraConfig:
    """Create a LoRA configuration for parameter-efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) adds small trainable matrices to the model
    instead of fine-tuning all parameters. This dramatically reduces:
    - Memory usage (only train a few million parameters instead of billions)
    - Training time (fewer parameters to update)
    - Storage space (adapters are typically <100MB vs GB for full models)
    
    Key hyperparameters:
    - r: Rank of the low-rank matrices (higher = more capacity but more parameters)
    - lora_alpha: Scaling factor (typically 2x rank for stable training)
    - lora_dropout: Regularization to prevent overfitting
    - target_modules: Which layers to apply LoRA to
    
    Returns:
        LoraConfig instance
    """
    return LoraConfig(
        # Rank of the low-rank matrices
        # Higher values (16-64) capture more complex patterns but use more memory
        # Lower values (4-8) are more parameter-efficient but may underfit
        r=16,
        # Scaling factor for LoRA updates
        # The effective learning rate is multiplied by alpha/r
        # Setting alpha=2*r is a common rule of thumb
        lora_alpha=32,
        # Dropout rate for regularization
        # 0.05 provides light regularization without hampering learning
        lora_dropout=0.05,
        # Don't train bias parameters separately
        # This simplifies the adapter while maintaining performance
        bias="none",
        # Specify this is a causal language model
        task_type="CAUSAL_LM",
        # Apply LoRA to attention layers
        # q_proj, k_proj, v_proj, o_proj are the query, key, value, and output projections
        # This targets the most important parts of the transformer
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main() -> None:
    """Main training entry point for SFT and DPO.
    
    This script handles both training stages:
    1. SFT (Supervised Fine-Tuning): Train on input-output pairs
    2. DPO (Direct Preference Optimization): Refine with preference pairs
    
    Usage:
        # Train SFT adapter
        python scripts/train.py --stage sft --data data/train.jsonl --output output/sft
        
        # Train DPO on top of SFT
        python scripts/train.py --stage dpo --data data/preferences.jsonl \\
            --sft-adapter output/sft/adapter --output output/dpo
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a QLoRA SFT adapter or DPO-tune an SFT adapter.")
    # Which training stage to run
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    # Base model to fine-tune
    # Qwen2.5-3B-Instruct is a good balance of size and capability
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    # Path to the SFT adapter (required for DPO stage)
    parser.add_argument("--sft-adapter", type=Path, help="Required for DPO; output from the SFT stage.")
    # Path to training data
    parser.add_argument("--data", type=Path, required=True, help="SFT or preference JSONL.")
    # Optional validation data for early stopping
    parser.add_argument("--validation-data", type=Path, help="Optional held-out validation JSONL for this training stage.")
    # Output directory for the trained adapter
    parser.add_argument("--output", type=Path, required=True)
    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    # Gradient accumulation simulates larger batch sizes
    # Effective batch size = batch_size * gradient_accumulation
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    # Maximum sequence length (includes both input and output)
    parser.add_argument("--max-length", type=int, default=1024)
    # Learning rate (typically higher for LoRA than full fine-tuning)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # Disable 4-bit quantization (requires more GPU memory)
    parser.add_argument("--no-4bit", action="store_true", help="Load full precision weights; needs substantially more GPU memory.")
    args = parser.parse_args()

    # Create the output directory
    args.output.mkdir(parents=True, exist_ok=True)
    # Determine whether to use quantization
    use_4bit = not args.no_4bit
    
    # Handle SFT training
    if args.stage == "sft":
        # Load the base model and tokenizer
        model, tokenizer = load_base(args.model, quantized=use_4bit)
        # Load and prepare training data
        rows = load_sft_rows(args.data, tokenizer)
        # Load validation data if provided
        validation_rows = (
            load_sft_rows(args.validation_data, tokenizer)
            if args.validation_data
            else None
        )
        # Wrap the model with LoRA adapters
        # This freezes the base model and adds trainable adapters
        model = get_peft_model(model, lora_config())
        
        # Configure the training run
        config = SFTConfig(
            output_dir=str(args.output),
            # Number of passes through the training data
            num_train_epochs=args.epochs,
            # Batch size per GPU
            per_device_train_batch_size=args.batch_size,
            # Simulate larger batch sizes with gradient accumulation
            gradient_accumulation_steps=args.gradient_accumulation,
            # Learning rate for the LoRA parameters
            learning_rate=args.learning_rate,
            # Log metrics every 10 steps for monitoring
            logging_steps=10,
            # Save a checkpoint after each epoch
            save_strategy="epoch",
            # Evaluate after each epoch if validation data is provided
            eval_strategy="epoch" if validation_rows else "no",
            # Enable gradient checkpointing to save memory
            # This trades compute for memory by recomputing activations during backward pass
            gradient_checkpointing=True,
            # Maximum sequence length
            max_seq_length=args.max_length,
            # Field name in the dataset that contains the text
            dataset_text_field="text",
            # Disable reporting to external services (can enable W&B, TensorBoard)
            report_to="none",
        )
        
        # Create the trainer
        trainer = SFTTrainer(
            model=model,
            args=config,
            # Convert the list of dicts to a Hugging Face Dataset
            train_dataset=Dataset.from_list(rows),
            eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
            # Pass the tokenizer for processing
            processing_class=tokenizer,
        )
    # Handle DPO training
    else:
        # DPO requires an SFT adapter as the starting point
        if args.sft_adapter is None:
            parser.error("--sft-adapter is required for the DPO stage.")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load preference pairs
        rows = load_preference_rows(args.data, tokenizer)
        validation_rows = (
            load_preference_rows(args.validation_data, tokenizer)
            if args.validation_data
            else None
        )
        
        # Load the base model
        base, _ = load_base(args.model, quantized=use_4bit)
        # Load the SFT adapter and set it to trainable mode
        model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
        
        # Configure DPO training
        config = DPOConfig(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            # Use a lower learning rate for DPO (10x smaller than SFT)
            # DPO is more sensitive to learning rate
            learning_rate=args.learning_rate / 10,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if validation_rows else "no",
            gradient_checkpointing=True,
            # Maximum total sequence length (prompt + chosen/rejected)
            max_length=args.max_length,
            # Maximum length for just the prompt
            max_prompt_length=args.max_length // 2,
            # Beta controls the strength of the KL constraint
            # Higher values = stay closer to the reference model
            beta=0.1,
            report_to="none",
        )
        
        # Create the DPO trainer
        trainer = DPOTrainer(
            model=model,
            # ref_model=None means use the model without adapter as reference
            ref_model=None,
            args=config,
            train_dataset=Dataset.from_list(rows),
            eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
            processing_class=tokenizer,
        )

    # Start training
    trainer.train()
    # Save the training state
    trainer.save_state()
    # Save the trained adapter
    trainer.save_model(str(args.output / "adapter"))
    # Save the tokenizer with the adapter
    tokenizer.save_pretrained(str(args.output / "adapter"))
    print(f"Saved {args.stage.upper()} adapter to {args.output / 'adapter'}")


if __name__ == "__main__":
    main()
