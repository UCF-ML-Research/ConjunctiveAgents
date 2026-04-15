from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_LOADED = {}

def load_generator(model_name: str, max_new_tokens: int = 64):
    key = (model_name, max_new_tokens)
    if key in _LOADED:
        return _LOADED[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    _LOADED[key] = gen
    return gen

