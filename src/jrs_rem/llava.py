"""Hugging Face LLaVA-1.5 adapter. No other target VLM is included."""
import torch
import transformers
from PIL import Image


def dtype_kwargs(dtype):
    keyword = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return {keyword: getattr(torch, dtype)}


def load_model(path, device="cuda", dtype="float16", revision=None):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(path, revision=revision)
    model = LlavaForConditionalGeneration.from_pretrained(
        path, revision=revision, **dtype_kwargs(dtype), low_cpu_mem_usage=True).to(device).eval()
    if getattr(model.config, "model_type", None) != "llava":
        raise ValueError("Only Hugging Face LLaVA is supported")
    return model, processor


def decoder_layers(model):
    for path in ("model.language_model.layers", "model.language_model.model.layers",
                 "language_model.model.layers", "model.layers"):
        obj = model
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
            if len(obj):
                return obj
        except (AttributeError, TypeError):
            continue
    raise ValueError("Unsupported LLaVA decoder layout")


def intervention_modules(model):
    """Match hidden_states[1:] including the final post-RMSNorm entry."""
    layers = decoder_layers(model)
    for module in model.modules():
        if getattr(module, "layers", None) is layers:
            norm = getattr(module, "norm", None)
            if norm is not None:
                return [*layers[:-1], norm]
    raise ValueError("Cannot locate final LLaVA language-model norm")


def build_input(processor, row, device, text_only=False, prefix=""):
    content = [{"type": "text", "text": prefix + row["txt"]}]
    image = None
    if row.get("img") and not text_only:
        with Image.open(row["img"]) as source:
            image = source.convert("RGB")
        content.append({"type": "image"})
    # Explicit processor call handles both supported Transformers major versions.
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=False)
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    return processor(**kwargs).to(device)


@torch.inference_mode()
def hidden_states(model, inputs):
    result = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    return torch.stack([h[0, -1].float() for h in result.hidden_states])


@torch.inference_mode()
def generate(model, processor, inputs, max_new_tokens):
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
