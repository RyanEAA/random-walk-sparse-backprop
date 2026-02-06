# randomwalk.py
"""
Random-walk sparse backprop (Option B, row-sparse backward) in ONE file.

What this file gives you:
- LinearExecutionTracer: traces the true execution order of nn.Linear layers via a forward hook pass
- RandomWalkRowSampler: generates, for each Linear layer, a set of selected OUTPUT ROW indices (neurons)
- SparseLinearFunction: custom autograd that computes grads ONLY for selected rows (plus full grad_x)
- SparseLinear: nn.Module wrapper around SparseLinearFunction
- SparseMLPWrapper: convenience wrapper to run a standard MLP with sparse-linear layers
- demo / sanity test: compare dense vs sparse grads on the selected rows

Important design choice (first milestone):
- We make backward sparse by selecting output rows I per layer.
- This reduces the dominant backward matmul sizes from (out x in) to (|I| x in).
- Heavy compute stays in torch kernels (matmul/index_select/index_copy_), not Python loops over weights.

This is meant to be a clean starting point. Once working, we can tighten to (I,J) edge-sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) Trace true execution order of Linear layers
# ----------------------------

class LinearExecutionTracer:
    """
    Records the actual execution order of nn.Linear modules during a forward pass.
    This avoids relying on model.named_modules() order, which isn't guaranteed to match forward order.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self.exec_linears: List[nn.Linear] = []

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(module, nn.Linear):
            self.exec_linears.append(module)

    @torch.no_grad()
    def trace(self, example_input: torch.Tensor) -> List[nn.Linear]:
        self.exec_linears = []
        self._handles = []

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._hook))

        _ = self.model(example_input)

        for h in self._handles:
            h.remove()
        self._handles = []

        # Deduplicate while preserving order (sometimes modules can be hit multiple times in odd graphs)
        seen = set()
        ordered_unique: List[nn.Linear] = []
        for layer in self.exec_linears:
            if id(layer) not in seen:
                seen.add(id(layer))
                ordered_unique.append(layer)
        return ordered_unique


# ----------------------------
# 2) Random-walk sampler that returns SELECTED OUTPUT ROWS per layer
# ----------------------------

@dataclass
class RandomWalkConfig:
    num_paths: int = 256                 # number of random walks per batch
    start_uniform: bool = True           # if True, choose starting output neuron uniformly
    epsilon_greedy: float = 0.0          # probability to choose uniformly instead of weight-biased transition
    weight_temperature: float = 1.0      # >1 flattens, <1 sharpens distribution when using weight magnitudes
    min_rows_per_layer: int = 1          # ensure at least this many rows are selected (fallback)
    device: Optional[torch.device] = None


class RandomWalkRowSampler:
    """
    For a stack of Linear layers in forward order [L1, L2, ..., Lk],
    do random walks from the output layer back to the input layer and return,
    for each layer, the set of SELECTED OUTPUT ROW indices.

    Row-sparse backward uses these selected rows I_l to compute gradients for W_l[I_l, :].
    """

    def __init__(self, config: RandomWalkConfig):
        self.cfg = config

    def _safe_probs_from_abs_weights(
        self,
        row_abs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Convert abs-weight row to a probability distribution with optional temperature.
        row_abs: [in_features] nonnegative
        """
        # Avoid all-zero rows; fall back to uniform.
        s = row_abs.sum()
        if not torch.isfinite(s) or s.item() <= 0.0:
            return torch.full_like(row_abs, 1.0 / row_abs.numel())

        # Temperature scaling on weights (on positives)
        if temperature != 1.0:
            # (w ** (1/temperature)) is a common "temperature" analog for nonnegative values
            # temperature > 1 -> flatter; temperature < 1 -> sharper
            pow_ = 1.0 / float(temperature)
            row_abs = row_abs.pow(pow_)

        probs = row_abs / (row_abs.sum() + 1e-12)
        return probs

    @torch.no_grad()
    def sample_selected_rows(self, linears_fwd: Sequence[nn.Linear]) -> List[torch.Tensor]:
        """
        Returns a list of LongTensors [I_1, I_2, ..., I_k] in forward order.
        Each I_l contains unique selected output-row indices for that layer.
        """
        assert len(linears_fwd) > 0, "Need at least one nn.Linear layer."

        device = self.cfg.device
        if device is None:
            # assume model weights are on desired device
            device = linears_fwd[0].weight.device

        # Build absolute weight matrices in backward-walk order: [Lk, ..., L1]
        linears_rev = list(reversed(linears_fwd))
        abs_weights_rev = [lin.weight.detach().abs() for lin in linears_rev]  # [out, in] each

        # Selected output rows per layer in reverse order, as Python sets initially (cheap)
        selected_rows_rev: List[List[int]] = [[] for _ in linears_rev]

        out_dim = abs_weights_rev[0].shape[0]
        P = int(self.cfg.num_paths)

        # If start_uniform, we start uniformly from output neurons; else you can implement other strategies later.
        # We'll keep it simple for RW-RANDOM: uniform starting row.
        start_rows = torch.randint(low=0, high=out_dim, size=(P,), device=device)

        # Walk each path. Depth is number of layers; that loop is tiny.
        for p in range(P):
            i = int(start_rows[p].item())  # current "output neuron index" at this layer in reverse-walk
            for layer_idx_rev, W_abs in enumerate(abs_weights_rev):
                out_features, in_features = W_abs.shape

                # clamp i to valid (safety; shouldn't be needed if dims match)
                if i < 0:
                    i = 0
                if i >= out_features:
                    i = out_features - 1

                # record that this output row i is selected for this layer
                selected_rows_rev[layer_idx_rev].append(i)

                # pick predecessor j for next layer (except after last reverse layer)
                # the predecessor index becomes the next i
                row_abs = W_abs[i]  # [in_features]
                if self.cfg.epsilon_greedy > 0.0 and torch.rand((), device=device).item() < self.cfg.epsilon_greedy:
                    j = int(torch.randint(0, in_features, (1,), device=device).item())
                else:
                    probs = self._safe_probs_from_abs_weights(row_abs, self.cfg.weight_temperature)
                    j = int(torch.multinomial(probs, 1).item())

                i = j  # move "back" to the previous layer's output index (i becomes j)

        # Convert to unique, sorted LongTensors, ensure minimum coverage
        selected_rows_fwd: List[torch.Tensor] = []
        for layer_rows in reversed(selected_rows_rev):  # back to forward order
            if len(layer_rows) == 0:
                # fallback: at least one row
                k_out = 1
                I = torch.randint(0, linears_fwd[len(selected_rows_fwd)].out_features, (k_out,), device=device)
                I = I.unique()
                selected_rows_fwd.append(I)
                continue

            I = torch.tensor(layer_rows, dtype=torch.long, device=device).unique()
            if I.numel() < self.cfg.min_rows_per_layer:
                out_features = linears_fwd[len(selected_rows_fwd)].out_features
                extra = torch.randint(0, out_features, (self.cfg.min_rows_per_layer,), device=device)
                I = torch.cat([I, extra]).unique()
            selected_rows_fwd.append(I)

        return selected_rows_fwd

class SparseReLUFunction(torch.autograd.Function):
    """
    ReLU forward is normal.
    Backward applies the correct ReLU derivative mask.
    Optionally also allows you to sparsify by zeroing grad outside selected indices,
    but for correctness it’s the derivative that matters.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # Save a boolean mask for backward
        mask = x > 0
        ctx.save_for_backward(mask)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (mask,) = ctx.saved_tensors
        return grad_out * mask


# ----------------------------
# 3) SparseLinear custom autograd (row-sparse backward)
# ----------------------------

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, rows_I: torch.Tensor):
        # forward: same as Linear
        y = x.matmul(W.t())
        if b is not None:
            y = y + b

        # save for backward
        ctx.save_for_backward(x, W, rows_I)
        ctx.has_bias = b is not None
        if b is not None:
            ctx.bias_shape = b.shape
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, W, rows_I = ctx.saved_tensors
        has_bias = ctx.has_bias

        # If no rows selected, return zeros
        if rows_I.numel() == 0:
            grad_x = torch.zeros_like(x)
            grad_W = torch.zeros_like(W)
            grad_b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype) if has_bias else None
            return grad_x, grad_W, grad_b, None

        # Select only chosen output columns from grad_out -> [B, k]
        grad_out_I = grad_out.index_select(dim=1, index=rows_I)

        # Select corresponding rows of W -> [k, in]
        W_I = W.index_select(dim=0, index=rows_I)

        # grad_x = grad_out_I @ W_I  -> [B, in]
        grad_x = grad_out_I.matmul(W_I)

        # grad_W rows only: grad_W_I = grad_out_I^T @ x -> [k, in]
        grad_W_I = grad_out_I.t().matmul(x)

        grad_W = torch.zeros_like(W)
        grad_W.index_copy_(0, rows_I, grad_W_I)

        grad_b = None
        if has_bias:
            grad_b_I = grad_out_I.sum(dim=0)
            grad_b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)
            grad_b.index_copy_(0, rows_I, grad_b_I)

        return grad_x, grad_W, grad_b, None



class SparseLinear(nn.Module):
    """
    Wraps an nn.Linear's parameters but uses SparseLinearFunction for forward/backward.
    Call: y = layer(x, rows_I)
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        # Share parameters with the original Linear
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor, rows_I: torch.Tensor) -> torch.Tensor:
        return SparseLinearFunction.apply(x, self.weight, self.bias, rows_I)


# ----------------------------
# 4) Convenience wrapper for a simple MLP (Linear/ReLU stacks)
# ----------------------------

class SparseMLPWrapper(nn.Module):
    """
    Wrap an existing MLP-like model with SparseLinear layers in the traced forward order.
    This wrapper assumes the model's forward is a simple sequential application of linears and activations.
    For complex models, you'd integrate SparseLinear into the model itself.

    Usage:
      wrapper = SparseMLPWrapper(model, example_input)
      logits = wrapper(x, selected_rows_per_layer)
    """

    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        super().__init__()
        self.model = model

        tracer = LinearExecutionTracer(model)
        self.linears_fwd: List[nn.Linear] = tracer.trace(example_input)

        if len(self.linears_fwd) == 0:
            raise ValueError("No nn.Linear layers were executed during the trace pass.")

        self.sparse_linears_fwd = nn.ModuleList([SparseLinear(lin) for lin in self.linears_fwd])

        # We'll also record a very simple "forward recipe" by re-tracing with hooks that record module outputs.
        # For a first milestone, we assume a standard MLP structure:
        #   x -> Linear -> ReLU -> Linear -> ReLU -> ... -> Linear -> logits
        #
        # If your model has different activations, you can edit `activation_fn` below or integrate directly.
        self.activation_fn = F.relu

        # Sanity: ensure dimensions line up as a chain
        for i in range(len(self.linears_fwd) - 1):
            if self.linears_fwd[i].out_features != self.linears_fwd[i + 1].in_features:
                raise ValueError(
                    "SparseMLPWrapper assumes a simple chain of Linear layers. "
                    f"Layer {i} out_features != layer {i+1} in_features "
                    f"({self.linears_fwd[i].out_features} != {self.linears_fwd[i+1].in_features})."
                )

    def forward(self, x: torch.Tensor, selected_rows_fwd: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(selected_rows_fwd) != len(self.sparse_linears_fwd):
            raise ValueError(
                f"Expected {len(self.sparse_linears_fwd)} row-index tensors, got {len(selected_rows_fwd)}."
            )

        h = x
        for idx, sparse_lin in enumerate(self.sparse_linears_fwd):
            h = sparse_lin(h, selected_rows_fwd[idx])
            if idx < len(self.sparse_linears_fwd) - 1:
                h = self.activation_fn(h)
        return h


# ----------------------------
# 5) Demo + correctness check utilities
# ----------------------------

def compare_dense_vs_sparse_grads_on_rows(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sampler: RandomWalkRowSampler,
    criterion=nn.CrossEntropyLoss(),
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Runs:
      - dense backward on original model
      - sparse backward on SparseMLPWrapper(model)
    and compares grad_W rows that were selected.

    Returns dict with stats and max diffs.

    NOTE: This expects an MLP-like chain model as described in SparseMLPWrapper.
    """
    device = x.device
    model = model.to(device)

    # ---- Trace and wrap
    wrapper = SparseMLPWrapper(model, example_input=x[:1])

    # ---- Sample selected rows for each linear layer
    linears_fwd = wrapper.linears_fwd
    selected_rows = sampler.sample_selected_rows(linears_fwd)

    # ---- Dense backward
    model.zero_grad(set_to_none=True)

    # Trace the true linears once (use wrapper's traced order)
    linears_fwd = wrapper.linears_fwd

    # Register forward hooks to capture each linear output tensor
    fwd_handles, saved_outputs, bwd_handles = _register_row_mask_hooks(linears_fwd, selected_rows)

    logits_dense = model(x)
    loss_dense = criterion(logits_dense, y)

    # Remove forward hooks (we captured outputs)
    for h in fwd_handles:
        h.remove()

    # Now attach backward hooks to the captured output tensors
    # Each hook masks grad_out columns to selected rows for that layer
    for li, out_tensor in enumerate(saved_outputs):
        I = selected_rows[li]

        def make_grad_hook(I_local):
            def grad_hook(grad):
                # grad shape: [B, out_features]
                mask = torch.zeros_like(grad)
                mask.index_fill_(1, I_local, 1.0)
                return grad * mask
            return grad_hook

        bwd_handles.append(out_tensor.register_hook(make_grad_hook(I)))

    loss_dense.backward()

    # cleanup bwd hooks
    for h in bwd_handles:
        h.remove()

    dense_grads_W = [lin.weight.grad.detach().clone() for lin in linears_fwd]
    dense_grads_b = [lin.bias.grad.detach().clone() if lin.bias is not None else None for lin in linears_fwd]


    # ---- Sparse backward
    model.zero_grad(set_to_none=True)
    logits_sparse = wrapper(x, selected_rows)
    loss_sparse = criterion(logits_sparse, y)
    loss_sparse.backward()

    sparse_grads_W = [lin.weight.grad.detach().clone() for lin in linears_fwd]
    sparse_grads_b = [lin.bias.grad.detach().clone() if lin.bias is not None else None for lin in linears_fwd]

    # ---- Compare only selected rows
    layer_reports = []
    ok = True
    for li, lin in enumerate(linears_fwd):
        I = selected_rows[li]
        if I.numel() == 0:
            layer_reports.append({"layer": li, "rows": 0, "ok": True, "max_abs_diff_W": 0.0})
            continue

        W_dense_I = dense_grads_W[li].index_select(0, I)
        W_sparse_I = sparse_grads_W[li].index_select(0, I)
        diff_W = (W_dense_I - W_sparse_I).abs()
        max_abs_diff_W = float(diff_W.max().item()) if diff_W.numel() else 0.0

        # Bias comparison on selected rows
        max_abs_diff_b = 0.0
        if lin.bias is not None and dense_grads_b[li] is not None and sparse_grads_b[li] is not None:
            b_dense_I = dense_grads_b[li].index_select(0, I)
            b_sparse_I = sparse_grads_b[li].index_select(0, I)
            diff_b = (b_dense_I - b_sparse_I).abs()
            max_abs_diff_b = float(diff_b.max().item()) if diff_b.numel() else 0.0

        layer_ok = torch.allclose(W_dense_I, W_sparse_I, atol=atol, rtol=rtol)
        if lin.bias is not None:
            layer_ok = layer_ok and torch.allclose(
                dense_grads_b[li].index_select(0, I),
                sparse_grads_b[li].index_select(0, I),
                atol=atol, rtol=rtol
            )

        ok = ok and layer_ok
        layer_reports.append({
            "layer": li,
            "out_features": lin.out_features,
            "in_features": lin.in_features,
            "selected_rows": int(I.numel()),
            "max_abs_diff_W": max_abs_diff_W,
            "max_abs_diff_b": max_abs_diff_b,
            "ok": bool(layer_ok),
        })

    return {
        "ok": ok,
        "loss_dense": float(loss_dense.item()),
        "loss_sparse": float(loss_sparse.item()),
        "selected_rows_per_layer": [int(I.numel()) for I in selected_rows],
        "layer_reports": layer_reports,
    }


# ----------------------------
# 6) Minimal example model (you can delete this in your repo and use your own)
# ----------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, h1=256, h2=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x):
        # Expect x: [B, 784]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------------
# 7) Quick run (sanity) when executed directly
# ----------------------------
def _register_row_mask_hooks(linears_fwd, selected_rows_fwd):
    """
    Registers hooks so that during backward, the gradient flowing OUT of each linear
    only keeps columns in selected_rows_fwd[layer_idx].
    Returns a list of hook handles (call .remove()).
    """
    handles = []
    saved_outputs = []

    # First, capture each linear's output tensor during forward
    def make_fwd_hook():
        def fwd_hook(module, inp, out):
            saved_outputs.append(out)
        return fwd_hook

    fwd_handles = []
    for lin in linears_fwd:
        fwd_handles.append(lin.register_forward_hook(make_fwd_hook()))

    # We return fwd_handles and will add backward hooks after a forward pass,
    # because we need the actual output tensors.
    return fwd_handles, saved_outputs, handles

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake MNIST-like batch
    B = 64
    x = torch.randn(B, 784, device=device)
    y = torch.randint(0, 10, (B,), device=device)

    model = SimpleMLP().to(device)

    cfg = RandomWalkConfig(
        num_paths=256,
        epsilon_greedy=0.05,          # small exploration so you don't get rich-get-richer immediately
        weight_temperature=1.0,
        min_rows_per_layer=4,
        device=device,
    )
    sampler = RandomWalkRowSampler(cfg)

    report = compare_dense_vs_sparse_grads_on_rows(
        model=model,
        x=x,
        y=y,
        sampler=sampler,
        atol=1e-5,
        rtol=1e-4,
    )

    print("OK:", report["ok"])
    print("dense loss:", report["loss_dense"], "sparse loss:", report["loss_sparse"])
    print("selected rows per layer:", report["selected_rows_per_layer"])
    for r in report["layer_reports"]:
        print(r)
