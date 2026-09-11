import torch
from torch import nn

from ._validation import tensor
from .model import Forecast


class LinearLastLayerAdapter:
    """Capture a standard final nn.Linear's input in one deterministic forward.

    Accepts model outputs [B,C] or [B,H,C]. Restores module training flags and
    removes its temporary hook even on failure. Not for concurrent calls on the
    same model, reused output layers, or postprocessed/DCT model outputs.
    """

    def __init__(self, model: nn.Module, last_layer: nn.Linear):
        if not isinstance(last_layer, nn.Linear) or last_layer not in list(model.modules()):
            raise ValueError("last_layer must be an nn.Linear belonging to model")
        self.model, self.last_layer = model, last_layer

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Forecast:
        captured = []
        flags = [(module, module.training) for module in self.model.modules()]
        handle = self.last_layer.register_forward_hook(
            lambda module, args, output: captured.append(
                (args[0].detach().clone(), output.detach().clone())
            )
        )
        try:
            self.model.eval()
            mean = self.model(x)
        finally:
            handle.remove()
            for module, flag in flags:
                module.training = flag
        if len(captured) != 1:
            raise ValueError("The final linear layer must run exactly once per model call")
        features, head_output = captured[0]
        if (
            not isinstance(mean, torch.Tensor)
            or mean.shape != head_output.shape
            or not torch.equal(mean, head_output)
        ):
            raise ValueError("Model must return its linear output unchanged; use a custom adapter")
        if mean.ndim == 2:
            mean, features = mean[:, None, :], features[:, None, :]
        tensor(mean, "mean [B,H,C]", 3)
        tensor(features, "features [B,H,P]", 3)
        if self.last_layer.bias is not None:
            features = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
        return Forecast(mean=mean, features=features)


def idct_design(coefficient_features: torch.Tensor, idct: torch.Tensor) -> torch.Tensor:
    """Time-domain design [B,H,P+1] for a biased linear layer shared over DCT bins.

    coefficient_features [B,K,P], truncated inverse DCT [H,K]. The transformed
    bias feature is the row SUM of iDCT, not an unconditional column of ones.
    The caller retains their exact preprocessing, offset and mean computation.
    """
    features = tensor(coefficient_features, "coefficient_features", 3)
    basis = tensor(idct, "idct", 2).to(features)
    if features.shape[1] != basis.shape[1]:
        raise ValueError("DCT coefficient dimensions do not match")
    transformed = torch.einsum("hk,bkp->bhp", basis, features)
    bias = basis.sum(-1)[None, :, None].expand(features.shape[0], -1, 1)
    return torch.cat([transformed, bias], dim=-1)
