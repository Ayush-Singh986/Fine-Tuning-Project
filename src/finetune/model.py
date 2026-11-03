# Import os for file path operations and environment variable access
import os
# Import Protocol for defining structural subtypes (duck typing with type checking)
from typing import Protocol


class Extractor(Protocol):
    """Protocol defining the interface for model inference.
    
    A Protocol in Python is like an interface in other languages.
    Any class that has a generate method with this signature is
    considered an Extractor, even if it doesn't explicitly inherit from this class.
    
    This allows us to swap different model implementations (Transformers, vLLM, etc.)
    without changing the calling code. The API and evaluation scripts only depend
    on this interface, not the concrete implementation.
    """
    
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the model given a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            The model's generated response as a string
        """
        ...


class TransformersExtractor:
    """Model inference using Hugging Face Transformers and PEFT for LoRA adapters.
    
    This class handles loading and running inference on fine-tuned models.
    It supports both:
    - Merged models: Base model with LoRA weights merged in
    - LoRA adapters: Separate adapter weights that must be applied to a base model
    
    The class automatically detects whether the model_path is an adapter or merged
    model and loads appropriately. It also handles device placement (CPU/GPU) and
    data type selection (fp32, fp16, bf16) based on available hardware.
    """
    
    def __init__(self, model_path: str, base_model: str | None = None) -> None:
        """Initialize the model and tokenizer.
        
        Args:
            model_path: Path to either a merged model or LoRA adapter directory
            base_model: Optional override for the base model (for adapters)
        """
        # Import heavy dependencies only when needed
        # This allows the library to be imported without GPU dependencies
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Model inference requires the inference extra: "
                "pip install -e '.[inference]'."
            ) from error

        # Check if model_path is a LoRA adapter by looking for adapter_config.json
        # This file is created by the PEFT library when saving adapters
        adapter_config = os.path.join(model_path, "adapter_config.json")
        is_adapter = os.path.isfile(adapter_config)
        
        # If it's an adapter, we need the base model path
        # The base model path can be provided explicitly or read from the adapter config
        adapter_base = (
            base_model or PeftConfig.from_pretrained(model_path).base_model_name_or_path
            if is_adapter
            else None
        )
        
        # Load the tokenizer
        # For adapters, use the base model's tokenizer
        # For merged models, use the model's own tokenizer
        tokenizer_path = adapter_base or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Choose the optimal data type based on available hardware
        # bfloat16 is preferred on modern GPUs (Ampere+) for numerical stability
        # float16 is used on older GPUs
        # float32 is used on CPU (no half precision support)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        
        # Load the base model
        # device_map="auto" automatically places the model on available devices (GPU/CPU)
        # This is essential for large models that don't fit in a single GPU
        model = AutoModelForCausalLM.from_pretrained(
            adapter_base or model_path,
            device_map="auto",
            torch_dtype=dtype,
        )
        
        # If this is an adapter, load and apply the LoRA weights
        if is_adapter:
            self.model = PeftModel.from_pretrained(model, model_path)
        else:
            self.model = model
        
        # Enable KV cache for faster generation
        # The cache stores previously computed key-value pairs to avoid recomputation
        self.model.config.use_cache = True
        # Set the model to evaluation mode
        # This disables dropout and other training-specific behaviors
        self.model.eval()
        # Store the device where the model is loaded (for input tensor placement)
        self.device = self.model.get_input_embeddings().weight.device

    @classmethod
    def from_environment(cls) -> "TransformersExtractor":
        """Create an extractor from environment variables.
        
        This is a convenience method for deployment scenarios where
        model paths are configured via environment variables.
        
        Required environment variables:
            MODEL_PATH: Path to the merged model or LoRA adapter
            
        Optional environment variables:
            BASE_MODEL: Override the base model path (for adapters)
            
        Returns:
            A TransformersExtractor instance
            
        Raises:
            RuntimeError: If MODEL_PATH is not set
        """
        model_path = os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("Set MODEL_PATH to a merged model or saved LoRA adapter.")
        return cls(model_path, os.environ.get("BASE_MODEL"))

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the model.
        
        This method handles the full inference pipeline:
        1. Format the conversation into a prompt
        2. Tokenize the prompt
        3. Generate tokens from the model
        4. Decode the generated tokens back to text
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            The model's generated response as a string
        """
        import torch

        # Use the tokenizer's chat template if available
        # Modern models (like Qwen, Llama 3) have chat templates built-in
        # These templates format the conversation in the model-specific way
        if self.tokenizer.chat_template:
            # Apply the chat template and prepare inputs for generation
            # add_generation_prompt=True adds the assistant prefix
            # return_tensors="pt" returns PyTorch tensors
            # return_dict=True returns a dict with input_ids, attention_mask, etc.
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        else:
            # Fall back to simple format for models without chat templates
            # This is a simple "role: content" format separated by newlines
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate using inference mode for efficiency
        # inference_mode is a context manager that disables gradient computation
        # This significantly reduces memory usage during generation
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                # Use greedy decoding for deterministic outputs
                # This is important for structured JSON extraction
                do_sample=False,
                # Limit the number of generated tokens
                # 512 tokens is sufficient for most invoice extractions
                max_new_tokens=512,
                # Use the EOS token as the pad token (required by some models)
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the newly generated tokens (not the input prompt)
        # output[0] is the first batch item
        # inputs["input_ids"].shape[1] is the length of the input
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        
        # Decode the tokens back to text
        # skip_special_tokens=True removes padding/EOS tokens from the output
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
