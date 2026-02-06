# RandomWalk Sparse Backpropagation

This repository explores **random-walk–based sparse backpropagation**, a training strategy that reduces the number of parameter updates during learning by selectively propagating gradients along random paths through the network.

The project evolved through **two distinct implementations**:

1. an *original dense-backward + gradient masking* approach, and
2. a **new true sparse-backward algorithm** implemented with custom autograd.

This README summarizes the algorithm, experimental findings, API, and current limitations.

---

## Motivation

Standard backpropagation updates *all* parameters at every step, which can be computationally expensive and unnecessary. Many updates may contribute little to learning.

**Goal:** reduce training cost by updating only a subset of parameters per step, while retaining competitive accuracy.

Random-walk sparse backprop addresses this by:

* sampling paths through the network via random walks on the weight graph
* updating only parameters visited by those paths

---

## Algorithm Overview

### Random-Walk Sparse Backprop (RW-Sparse)

At each training step:

1. Perform a forward pass as usual.
2. Starting from the output layer, sample `num_paths` random walks backward through the network.
3. Each walk selects a sequence of neurons (rows of weight matrices).
4. During backpropagation, **only the parameters corresponding to the visited rows receive gradients**.

This implementation focuses on **row-sparse backpropagation for `nn.Linear` layers**, which already yields substantial reductions in parameter updates.

---

## Original Implementation vs New Algorithm

### Original Implementation (Dense Backward + Masking)

* Uses standard PyTorch backpropagation.
* Computes *full dense gradients* for all parameters.
* Applies a mask afterward to zero out gradients for unvisited parameters.
* **Limitation:** compute cost is still dominated by dense backward passes.

### New Algorithm: True Sparse Backward (Option B)

* Uses **custom autograd functions** (`SparseLinearFunction`).
* Gradients are computed *only* for selected rows during backward.
* Backward matrix multiplications are reduced in size.
* Produces the **same gradients** as dense backprop with identical row-masking constraints.

**Key difference:**

> The new algorithm reduces *both* parameter updates **and** backward compute, not just updates.

---

## Experimental Results (MNIST)

Experiments were run with:

* FULL backpropagation
* RW-Sparse with 10, 50, and 200 paths

### 1. Accuracy vs Training Time

![Accuracy vs Traing Time Graph](/plots/acc_vs_time.png)

* FULL converges fastest in wall-clock time.
* RW-Sparse converges more slowly, especially with fewer paths.
* Increasing `num_paths` improves convergence speed and final accuracy.

**Interpretation:** RW-Sparse trades gradient strength for reduced updates.

---

### 2. Accuracy vs Cumulative Parameter Updates (Key Result)

![Accuracy vs Cumulative Parameter Updates](/plots/acc_vs_updates.png)


This is the most important metric.

Findings:

* RW-Sparse reaches competitive accuracy with **significantly fewer parameter updates**.
* With 50–200 paths, RW-Sparse is **more update-efficient** than FULL.

> Per parameter update, RW-Sparse extracts more learning signal.

This supports the core claim of the method.

---

### 3. Speed (Average Step Time)

![Accuracy vs Cumulative Parameter Updates](/plots/speed_bars.png)

* FULL: fastest per step
* RW-Sparse: slower per step, increasing with `num_paths`

**Reason:**

* random-walk sampling
* index selection and gradient scattering
* Python-level control overhead

This is an implementation limitation, not an algorithmic flaw.

---

## API Overview

The current research-oriented API is built around the following components:

```python
from randomwalk import (
    RandomWalkConfig,
    RandomWalkRowSampler,
    SparseMLPWrapper,
)
```

### Typical Usage (Sparse Backward)

```python
sampler = RandomWalkRowSampler(cfg)
selected_rows = sampler.sample_selected_rows(wrapper.linears_fwd)

logits = wrapper(x, selected_rows)
loss.backward()
optimizer.step()
```

### FULL Baseline

```python
logits = model(x)
loss.backward()
optimizer.step()
```

### Experiment Driver

All experiments are run via:

```bash
python run_experiments.py --dataset mnist --num-paths 10,50,200
```

Results are logged to CSV and plotted separately.

---

## Clean-API Direction (Planned)

A more user-friendly API is planned, e.g.:

```python
trainer = RandomWalkTrainer(
    model,
    mode="rw_sparse",
    num_paths=50,
)
trainer.fit(train_loader, test_loader)
```

This would abstract away tracing, sampling, and sparse backward details.

---

## Current Limitations

* Implemented only for `nn.Linear` layers
* Python overhead dominates runtime for small models
* Random walk sampling not yet vectorized
* No support yet for:

  * convolutional layers
  * attention / transformers
  * normalization layers
* Experiments currently limited to small-scale models (MNIST)

---

## Future Work

Planned directions include:

1. **Performance optimizations**

   * vectorized random-walk sampling
   * TorchScript / `torch.compile`
   * lower-level (C++/CUDA) kernels

2. **Architectural extensions**

   * apply RW-Sparse to classifier heads of CNNs
   * extend to transformer MLP blocks
   * explore sparse backward rules for Conv2D

3. **Algorithmic variants**

   * adaptive or learned path sampling
   * importance-weighted random walks
   * hybrid dense/sparse schedules

4. **Stronger evaluation**

   * multi-seed runs with confidence intervals
   * larger datasets (CIFAR-10/100)
   * comparison with other sparse-gradient methods

---

## Summary

* RW-Sparse backprop significantly improves **update efficiency**.
* The new sparse-backward implementation is mathematically correct and more compute-aware than gradient masking.
* While current Python overhead limits wall-clock gains, results strongly motivate optimized implementations.

This repository serves as both a **research prototype** and a foundation for future sparse-training systems.


## Related Work

For sparse gradient masking approaches, see:

- Frankle & Carbin (2019), Lottery Ticket Hypothesis  
- Evci et al. (2020), RigL  
- Mohtashami (2022), Masked Training of Neural Networks with Partial Gradients  
- Jaiswal (2022), Training Your Sparse Neural Network Better with Any Mask
