import os
from typing import Protocol


class Extractor(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class TransformersExtractor:
    def __init__(self, model_path: str, base_model: str | None = None) -> None:
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Model inference requires the inference extra: "
                "pip install -e '.[inference]'."
            ) from error

        adapter_config = os.path.join(model_path, "adapter_config.json")
        is_adapter = os.path.isfile(adapter_config)
        adapter_base = (
            base_model or PeftConfig.from_pretrained(model_path).base_model_name_or_path
            if is_adapter
            else None
        )
        tokenizer_path = adapter_base or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            adapter_base or model_path,
            device_map="auto",
            torch_dtype=dtype,
        )
        if is_adapter:
            self.model = PeftModel.from_pretrained(model, model_path)
        else:
            self.model = model
        self.model.config.use_cache = True
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device

    @classmethod
    def from_environment(cls) -> "TransformersExtractor":
        model_path = os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("Set MODEL_PATH to a merged model or saved LoRA adapter.")
        return cls(model_path, os.environ.get("BASE_MODEL"))

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        else:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
