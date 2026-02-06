# rw_sparse/masker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable

import torch
import torch.nn as nn


@dataclass
class MaskStats:
    """Useful metrics to log each step/epoch."""
    density: float                 # fraction of trainable params with non-zero grads after masking
    num_params_total: int          # total trainable scalar params considered
    num_params_kept: int           # number kept (nonzero gradient entries)
    per_layer_density: Dict[str, float]


class RandomWalkMasker:
    """
    Random-walk sparse gradient masker for feedforward MLPs (Linear layers).

    Typical usage:
        masker = RandomWalkMasker(mode="rw_target", num_paths=200)
        ...
        loss.backward()                 # dense grads first
        stats = masker.apply(model, labels=y)
        optimizer.step()

    Modes:
        - "rw_random": start each walk at a random output class index
        - "rw_target": start each walk at the true class of a randomly sampled batch example
        - "full": no masking (debug / baseline)
    """

    def __init__(
        self,
        mode: str = "rw_target",
        num_paths: int = 200,
        *,
        device: Optional[torch.device] = None,
        include_bias: bool = True,
        layer_selector: str = "linear",  # future: "manual"
        eps: float = 1e-8,
    ):
        self.mode = mode
        self.num_paths = int(num_paths)
        self.device = device
        self.include_bias = include_bias
        self.layer_selector = layer_selector
        self.eps = eps

        # cached layer references discovered from the model
        self._layers: List[Tuple[str, nn.Linear]] = []
        # for optional caching later
        self._last_model_id: Optional[int] = None

    # -----------------------------
    # Public API
    # -----------------------------

    def apply(self, model: nn.Module, labels: Optional[torch.Tensor] = None) -> MaskStats:
        """
        Apply random-walk masking to gradients *in-place*.

        Expects:
          - model has already run forward pass
          - loss.backward() already called -> dense grads exist
        """
        # 0) handle "full" mode
        if self.mode == "full":
            return self._compute_density_from_grads(model)

        # 1) discover layers (only once per model instance, unless changed)
        self._maybe_discover_layers(model)

        # 2) build masks for each weight matrix in [fc1, fc2, ..., fclast] order
        rw_masks, rows_visited = self._build_walk_masks(model, labels=labels)

        # 3) apply mask to gradients (weights and optionally bias)
        self._apply_masks_to_grads(model, rw_masks, rows_visited)

        # 4) return density stats (measured after masking)
        return self._compute_density_from_grads(model)

    def set_num_paths(self, num_paths: int) -> None:
        self.num_paths = int(num_paths)

    # -----------------------------
    # Layer discovery
    # -----------------------------

    def _maybe_discover_layers(self, model: nn.Module) -> None:
        model_id = id(model)
        if self._last_model_id == model_id and self._layers:
            return

        self._layers = self._discover_linear_layers(model)
        if len(self._layers) < 2:
            raise ValueError(
                "RandomWalkMasker expects at least 2 nn.Linear layers "
                "(e.g., an MLP). Found fewer."
            )
        self._last_model_id = model_id

    def _discover_linear_layers(self, model: nn.Module) -> List[Tuple[str, nn.Linear]]:
        """
        Return all nn.Linear layers in forward order.
        For typical MLPs this matches input->hidden->output order.
        """
        layers: List[Tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((name, module))
        # named_modules returns nested order; for Sequential it's fine.
        # If you need strict forward order later, you can require the user
        # to pass a list of layer names.
        return layers

    # -----------------------------
    # Mask construction
    # -----------------------------

    def _build_walk_masks(
        self, model: nn.Module, labels: Optional[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
          rw_masks: list of masks for weights in forward order [L1, L2, ..., Llast]
          rows_visited: list of 1D tensors marking visited output rows for each layer
        """
        device = self._resolve_device(model)

        # get weights in *reverse* layer order for walking from output back to input
        layers_rev = list(reversed(self._layers))  # [(name_last, last_layer), ...]
        Ws = [layer.weight.detach().abs().to(device) for _, layer in layers_rev]

        # init masks (same shape as each W)
        rw_masks_rev = [torch.zeros_like(W, device=device) for W in Ws]
        rows_visited_rev = [torch.zeros(W.size(0), device=device) for W in Ws]

        out_dim = Ws[0].size(0)  # last layer output dimension (#classes)

        # Sanity: labels required for rw_target
        if self.mode in ("rw_target", "rw_target_strict") and labels is None:
            raise ValueError(f"mode='{self.mode}' requires labels")

        # ---- main loop over paths ----
        for _ in range(self.num_paths):
            i = self._choose_start_output(out_dim, labels=labels, device=device)
            for layer_idx, W in enumerate(Ws):
                row = W[i]  # [in_features] for that output neuron

                # build probability distribution over previous-layer neurons
                s = row.sum()
                if s <= self.eps:
                    probs = torch.ones_like(row) / row.numel()
                else:
                    probs = row / (s + self.eps)

                # sample previous neuron index
                j = torch.multinomial(probs, num_samples=1).item()

                # mark edge i -> j as active
                rw_masks_rev[layer_idx][i, j] = 1.0
                rows_visited_rev[layer_idx][i] = 1.0

                # step to previous layer
                i = j

        # reverse back to forward layer order
        rw_masks = list(reversed(rw_masks_rev))
        rows_visited = list(reversed(rows_visited_rev))
        return rw_masks, rows_visited

    def _choose_start_output(
        self,
        out_dim: int,
        labels: Optional[torch.Tensor],
        device: torch.device,
    ) -> int:
        """
        Choose starting output neuron index.
        """
        if self.mode == "rw_random":
            return int(torch.randint(low=0, high=out_dim, size=(1,), device=device).item())

        if self.mode == "rw_target":
            # pick a random sample from the batch and use its true label
            b = int(torch.randint(low=0, high=labels.size(0), size=(1,), device=device).item())
            i = int(labels[b].item())
            return max(0, min(i, out_dim - 1))

        raise ValueError(f"Unsupported mode: {self.mode}")

    # -----------------------------
    # Apply mask to gradients
    # -----------------------------

    def _apply_masks_to_grads(
        self,
        model: nn.Module,
        rw_masks: List[torch.Tensor],
        rows_visited: List[torch.Tensor],
    ) -> None:
        """
        Multiply parameter gradients by masks (in-place).
        Assumes rw_masks is in forward order corresponding to self._layers.
        """
        for (name, layer), mask in zip(self._layers, rw_masks):
            if layer.weight.grad is not None:
                # Ensure mask lives on same device as grad
                m = mask.to(layer.weight.grad.device)
                layer.weight.grad.mul_(m)

            if self.include_bias and layer.bias is not None and layer.bias.grad is not None:
                # Simple heuristic: only allow bias updates for rows that were visited
                # rows_visited entry corresponds to output neurons of this layer
                # We need the matching rows_visited tensor for this layer.
                # NOTE: zip above doesn't include rows_visited; align separately.
                pass

        # Bias masking, aligned properly:
        if self.include_bias:
            for (name, layer), rv in zip(self._layers, rows_visited):
                if layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad.mul_(rv.to(layer.bias.grad.device))

    # -----------------------------
    # Density measurement
    # -----------------------------

    def _compute_density_from_grads(self, model: nn.Module) -> MaskStats:
        """
        Compute fraction of trainable scalar parameters with non-zero gradients.
        This is your "density" metric after masking.
        """
        total = 0
        kept = 0
        per_layer: Dict[str, float] = {}

        for name, layer in self._layers if self._layers else self._discover_linear_layers(model):
            # weights
            if layer.weight.requires_grad and layer.weight.grad is not None:
                g = layer.weight.grad
                total_w = g.numel()
                kept_w = int((g != 0).sum().item())
                total += total_w
                kept += kept_w

            # biases
            if self.include_bias and layer.bias is not None and layer.bias.requires_grad and layer.bias.grad is not None:
                gb = layer.bias.grad
                total_b = gb.numel()
                kept_b = int((gb != 0).sum().item())
                total += total_b
                kept += kept_b

            # layer density (weights only, for clearer debugging)
            if layer.weight.grad is not None:
                g = layer.weight.grad
                per_layer[name] = float((g != 0).float().mean().item())
            else:
                per_layer[name] = 0.0

        density = (kept / total) if total > 0 else 0.0
        return MaskStats(
            density=float(density),
            num_params_total=int(total),
            num_params_kept=int(kept),
            per_layer_density=per_layer,
        )

    # -----------------------------
    # Helpers
    # -----------------------------

    def _resolve_device(self, model: nn.Module) -> torch.device:
        if self.device is not None:
            return self.device
        # infer from first parameter
        p = next(model.parameters(), None)
        return p.device if p is not None else torch.device("cpu")
