"""Last-prompt-token intervention for JRS-Rem defense."""
import torch
from .llava import decoder_layers, intervention_modules


def correction(text_hs, multi_hs, ds, threshold=0.2, mode="signed"):
    if mode != "signed":
        raise ValueError("Only signed projection is supported; regenerate legacy runs")
    force = multi_hs.float() - text_hs.float()
    ds = ds.float()
    ds = ds / ds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scalar = (force * ds).sum(dim=-1)
    magnitude = force.norm(dim=-1)
    cosine = scalar / magnitude.clamp_min(1e-8)
    vector = scalar.unsqueeze(-1) * ds
    return vector, (magnitude > 0) & (cosine >= threshold)


class Intervention:
    """Apply each decoder hook once during generation prefill, then leave decoding alone.

    Entry 0 is not edited. Intermediate entries use decoder outputs;
    the final entry uses the final norm output, matching HF hidden_states.
    Context-manager lifetime guarantees removal after failed generation as well.
    """
    def __init__(self, model, delta, active):
        self.layers = decoder_layers(model)
        if delta.ndim != 2 or len(delta) != len(self.layers) + 1 or len(active) != len(delta):
            raise ValueError("Direction must have embedding+decoder layers, matching this model")
        self.modules = intervention_modules(model)
        self.delta, self.active = delta, active
        self.handles, self.seen = [], set()

    def hook(self, index):
        def apply(module, inputs, output):
            if index in self.seen:
                return output
            self.seen.add(index)
            if not bool(self.active[index]):
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[0] != 1 or h.shape[-1] != self.delta.shape[-1]:
                raise ValueError("Intervention supports one sample at a time with matching hidden width")
            h = h.clone()
            h[0, -1] += self.delta[index].to(device=h.device, dtype=h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return apply

    def __enter__(self):
        try:
            for i, layer in enumerate(self.modules):
                self.handles.append(layer.register_forward_hook(self.hook(i + 1)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []
