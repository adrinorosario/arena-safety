# JAX Roadmap (parallel to ARENA)

Nov 4, 2026 · @Adrino

## Target, assumptions, setup

The target is to write, train and inspect a small transformer in JAX as fluently as in PyTorch, then reason about scaling it. The roadmap starts once ARENA 0.1 is done and runs alongside ARENA. Each JAX stage re-implements something ARENA has just had you build in PyTorch.

| Assumption | If wrong |
| --- | --- |
| PyTorch stays primary; JAX is a second framework, not a migration | Stages 4-6 would expand and ARENA would slow |
| PyTorch roadmap readiness gate is passed before Stage 1 | Go back to it; JAX fluency rests on the same shape reasoning |
| JAX runs on the M2 Air CPU, with Colab for accelerators | See compute below |
| Neural-net library is Flax NNX, with Equinox as a comparison | Swap Stage 4 if a lab you target uses another library |

**Compute on the M2 Air.** `pip install jax` gives the CPU backend on macOS ([JAX quickstart](https://melissawm.github.io/jax/notebooks/thinking_in_jax.html)), which is enough for Stages 1-5. Apple's `jax-metal` GPU plugin is experimental ([Apple](https://developer.apple.com/metal/jax/)). One forum user reports the JAX maintainers closed Metal issues in December 2025, citing no active development ([Apple Developer Forums](https://origin-devforums.apple.com/forums/thread/815139)). Third-party Metal backends exist in beta, e.g. `metaljax`, which needs macOS 14+, Python 3.12+ and has no float64 ([PyPI](https://pypi.org/project/metaljax/)). Use Colab for anything GPU- or TPU-dependent instead of debugging a beta backend.

- [ ] Open question: what is driving JAX for you (a lab, a codebase, TPU access, or general breadth)? The answer decides whether Stage 6 is core or optional.

## PyTorch-to-JAX mental model

JAX is not PyTorch with different names. Its transformations (`grad`, `jit`, `vmap`) work only on functionally pure Python functions, where all inputs arrive as arguments and all results leave as return values ([Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)). Most of the table below follows from that one rule.

| Concept | PyTorch | JAX | Why the difference |
| --- | --- | --- | --- |
| Arrays | Mutable tensors, `x[i] = v` | Immutable; `x = x.at[i].set(v)` returns a new array | Mutation makes program analysis and transformation hard |
| Gradients | `.backward()` fills `.grad` on leaf tensors | `jax.grad(f)` returns a new function that computes gradients | Differentiation is a function transformation, not a side effect |
| Randomness | Global stateful generator | Explicit keys, `jax.random.split` | Stateless keys stay reproducible under `jit` and parallelism |
| Batching | Write batched code by hand | Write per-example code, then `jax.vmap` | Vectorisation is a transformation |
| Speed | Eager by default; `torch.compile` optional | `jax.jit` traces to XLA; shapes must be static | Tracing records operations on abstract values, not data |
| Control flow | Python `if`/`for` on tensor values | `lax.cond`, `lax.scan`, `lax.fori_loop` under `jit` | A traced value has no concrete truth value |
| Model state | `nn.Module` owns parameters | Parameters are pytrees passed in and out; Flax NNX and Equinox wrap this | Equinox's authors frame this as the OO-vs-functional tension ([Kidger & Garcia](https://arxiv.org/abs/2111.00254)) |
| Optimizer | `optimizer.step()` mutates params | Optax `init`/`update` return new state; `apply_updates` returns new params ([Optax](https://optax.readthedocs.io/en/latest/getting_started.html)) | Same functional rule |

One JAX user argues that comparing JAX with deep-learning frameworks misleads, because its real focus is composable program transformations ([Eric Ma](https://ericmjl.github.io/notes/differential-computing-with-jax)). Flax NNX re-introduces PyTorch-style stateful modules on top of that core ([Flax](https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html)), so it is easy to write PyTorch-shaped JAX without learning the core. Stages 1-3 deliberately use no NN library for that reason.

Checkpoint question: if `jit` traces a function once with abstract values, what happens to a `print(x)` inside it on the second call?

## Working method: translation drills

The core drill is translation. Take something you just wrote in PyTorch for ARENA and rewrite it in JAX, then check that the two agree numerically. This uses ARENA as a supply of solved, tested problems, so JAX time never needs new problem design.

1. **Same shape contract.** Copy the PyTorch shape contract unchanged; shapes do not change between frameworks.
2. **Pure core first.** Write the computation as a pure function `f(params, x) -> y`, with no library, no globals, no mutation.
3. **Numerical parity test.** Convert the same inputs and weights with `np.asarray`, run both versions, and require `np.allclose(torch_out, jax_out, atol=1e-5)`. Do the same for gradients.
4. **Transform last.** Add `vmap`, then `jit`, only after parity holds. Re-run the parity test after each.
5. **Lookup ledger, JAX edition.** Log each lookup with the PyTorch equivalent beside it. Repeats go on a two-column op card (PyTorch | JAX).
6. **Sharp-bit log.** Each time JAX behaves differently than you expected, write one line: what you expected, what happened, which rule caused it.

The sharp-bit log is the main fluency signal. It should fill fast in Stages 1-3, then go quiet.

Checkpoint question: why compare *gradients* across frameworks, not just outputs? Which bug class does that catch that output parity misses?

## Stage 1: jax.numpy, immutability, PRNG, sharp bits

Start Stage 1 right after ARENA 0.1. Its translation target is your own ray tracer, which is pure array code with no learning.

- [ ] **Quickstart: How to think in JAX** ([JAX](https://melissawm.github.io/jax/notebooks/thinking_in_jax.html)). Covers `jax.numpy` as a NumPy-like interface, immutability, and an overview of the transformations.
- [ ] **The Sharp Bits** ([JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)). Read it twice: once now, once after Stage 3. Topics:
  - pure functions, and what `jit` does to side effects and globals
  - out-of-place updates via `.at[idx].set/add/...`
  - out-of-bounds indexing, which is clamped or skipped instead of raising
  - random numbers, control flow, dynamic shapes, `float32` default
- [ ] **JAX 101: Pseudorandom numbers** ([JAX 101 index](https://docs.jax.dev/en/latest/_sources/jax-101.rst)). Keys, `split`, `fold_in`; why reusing a key is a bug.
- [ ] **UvA Tutorial 2 (JAX): Introduction to JAX+Flax, JAX half only** ([UvA](https://uvadlc-notebooks.readthedocs.io/)). The PyTorch version of the same notebook is in your PyTorch roadmap, so diff the two.
- [ ] **Translation drill: ARENA 0.1 ray tracing** in `jax.numpy`, with a parity test against your PyTorch solution.

Exit test, no lookups:

- Replace `x[mask] = 0` with an out-of-place equivalent, and explain why the boolean-mask version fails under `jit`.
- Generate three independent Gaussian batches from one seed, correctly split, reproducibly.
- Predict the output of `jnp.arange(5)[10]` and justify it from the Sharp Bits.

Checkpoint question: NumPy's global RNG works fine in single-threaded code. What specifically breaks when that design meets `vmap` or multiple devices?

## Stage 2: transformations: grad, jit, vmap

This is the core of JAX. Spend the most time here, paired with ARENA 0.2 and 0.3.

- [ ] **JAX 101: Just-in-time compilation, Automatic vectorization, Automatic differentiation**, read in that order ([JAX 101](https://docs.jax.dev/en/latest/_sources/jax-101.rst)). Key ideas: tracing, `static_argnums`, recompilation on new shapes, `in_axes`/`out_axes`, `value_and_grad`, `has_aux`.
- [ ] **The Autodiff Cookbook** and **Advanced automatic differentiation**: higher-order derivatives, forward vs reverse mode (`jvp`, `vjp`), Jacobians and Hessians ([advanced guides index](https://docs.jax.dev/en/latest/export/index.html)).
- [ ] **Translation drill A: ARENA 0.2.** A pure-function MLP and conv layer on MNIST in raw JAX, with params as a dict, `jax.grad` and a hand-written SGD update. Parity-test the gradients against PyTorch.
- [ ] **Translation drill B: ARENA 0.3.** Re-implement your SGD, RMSprop and Adam as pure functions `(params, grads, state) -> (params, state)`.

Exit test, no lookups:

- **Per-example gradients** for a batch via `vmap(grad(loss), in_axes=(None, 0, 0))`. State why this is awkward in plain PyTorch.
- A Hessian-vector product with `jvp` of `grad`, without forming the Hessian.
- Time a training step with and without `jit`. Use `block_until_ready()` so you measure compute, not dispatch.
- Make a jitted function recompile on purpose, then stop it from doing so.

Checkpoint question: `jax.grad` returns a function, while `.backward()` mutates tensors. Which design makes `grad(grad(f))` trivial, and why?

## Stage 3: pytrees, control flow, scan, state

Pytrees are how JAX carries model parameters and optimizer state; `lax` control flow is how loops survive `jit`. Pair this stage with ARENA 0.4 (backprop).

- [ ] **JAX 101: Pytrees, Control flow and logical operators with JIT, Tracing, Stateful computations** ([JAX 101](https://docs.jax.dev/en/latest/_sources/jax-101.rst)). Key ideas:
  - `jax.tree.map`, flatten/unflatten, custom pytree nodes
  - `lax.cond`, `lax.switch`, `lax.fori_loop`, `lax.while_loop`, `lax.scan`
  - why Python `if x > 0` fails on a traced value
  - threading state through pure functions as `(state, x) -> (state, y)`
- [ ] **Introduction to debugging**: `jax.debug.print` and compiled breakpoints, and `checkify` ([guides index](https://docs.jax.dev/en/latest/export/index.html)).
- [ ] **Translation drill: ARENA 0.4.** Compare your PyTorch autograd engine with `jax.grad` on the same composite function, then write the training loop as a single `lax.scan` over steps.
- [ ] **Re-read the Sharp Bits.** Note which items now seem obvious and which still surprise you.

Exit test, no lookups:

- An RNN forward pass over `(seq, batch, d)` with `lax.scan`, jitted and differentiated.
- Adam implemented with its state as a pytree mirroring params, using `jax.tree.map` throughout.
- Count the parameters of a nested params dict in one line.

Checkpoint question: why does `lax.scan` compile once regardless of sequence length, while a Python `for` loop under `jit` unrolls? What does that cost at compile time?

## Stage 4: neural nets: Flax NNX, Optax, and an interp toolkit

Write the ARENA 1.1 transformer in JAX here. Use Flax NNX as the main library; Flax's docs direct new users to NNX over the older Linen API ([Flax](https://flax.readthedocs.io/en/stable/_sources/index.rst)).

- [ ] **Flax NNX basics**, **MNIST tutorial**, and **Why Flax NNX?** ([Flax docs](https://flax.readthedocs.io/en/v0.12.0/search.html)). Key ideas:
  - NNX modules own their variables and create them eagerly at construction
  - Linen modules are stateless and lazy
  - `nnx.jit` is NNX's version of `jax.jit` ([Linen to NNX](https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html))
- [ ] **Optax: Getting started** ([Optax](https://optax.readthedocs.io/en/latest/getting_started.html)). A `GradientTransformation` is an `init`/`update` pair. Build optimizers with `optax.chain`, e.g. global-norm clipping then AdamW with a schedule.
- [ ] **Equinox, for contrast** ([paper](https://arxiv.org/abs/2111.00254)). Its models are plain pytrees plus filtered transformations. Port one MLP to see the alternative to NNX's design.
- [ ] **DeepLearning.AI, "Build and Train an LLM with JAX"** ([course](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax)). A short Google-partnered course: a \~20M-parameter MiniGPT with Flax NNX, Grain for data, Optax, and Orbax checkpointing.
- [ ] **UvA JAX Tutorial 6: Transformers and Multi-Head Attention** ([UvA](https://uvadlc-notebooks.readthedocs.io/)). Check which Flax API it uses before following it line by line; the notebooks were ported from PyTorch earlier.
- [ ] **Translation drill: ARENA 1.1.** Port your transformer to Flax NNX, load identical weights, and parity-test the logits.
- [ ] **Interp in JAX (pair with ARENA 1.2): Penzai + Treescope** ([Penzai](https://penzai.readthedocs.io/en/latest/_sources/index.rst)). Google DeepMind's library treats models as editable pytree data, with named axes and a transformer implementation for interpretability and model surgery. Its paper appeared at the ICML 2024 Mechanistic Interpretability workshop ([Johnson](https://arxiv.org/html/2408.00211v1)). Redo one induction-head analysis in Penzai.

Exit test, no lookups:

- A Flax NNX transformer block plus Optax AdamW training step, jitted, from a blank file.
- Save and restore a checkpoint, then resume with identical loss.
- Capture one layer's activations in NNX (e.g. `sow`), and in Penzai.

Checkpoint question: NNX re-adds mutable modules on top of a pure core. Where does the purity come back when you call `nnx.jit`?

## Stage 5: internals: jaxprs, custom derivatives, Autodidax

This stage turns JAX from a black box into a system you can predict. It is the JAX counterpart to the micrograd step in the PyTorch roadmap. Run it alongside the ARENA Chapter 1 branches.

- [ ] **JAX 601: The jaxpr language** and **Primitives** ([JAX 601](https://docs.jax.dev/en/latest/601/index.html)). Read `jax.make_jaxpr(f)(x)` output fluently, and see what each transformation does to a jaxpr.
- [ ] **Custom derivative rules** with `custom_jvp`/`custom_vjp`, and **jax.checkpoint** (remat) to control saved activations ([guides index](https://docs.jax.dev/en/latest/export/index.html)).
- [ ] **Autodidax: JAX core from scratch.** Build tracing, jaxprs, autodiff and `jit` in pure Python, one layer at a time; Part 1 treats transformations as interpreters ([JAX 601](https://docs.jax.dev/en/latest/601/index.html)).
- [ ] **Autodidax2, part 1**: a newer rebuild reflecting current JAX internals. Do it after Autodidax.
- [ ] **Writing custom jaxpr interpreters**, e.g. one that counts FLOPs or logs every intermediate.

Exit test, no lookups:

- From a printed jaxpr of a 2-layer MLP loss, identify every primitive and predict the jaxpr of its `grad`.
- A numerically stable `logsumexp` with a `custom_jvp`, parity-tested against `jax.nn.logsumexp` gradients.
- A jaxpr interpreter that records every intermediate's shape: a JAX-native activation cache.

Checkpoint question: Autodidax treats `jvp` and `vmap` as interpreters over the same primitives. Why does that make transformations compose (e.g. `vmap(grad(f))`) without special cases?

## Stage 6: scale: sharding, profiling, the scaling book

This is where JAX's advantage over PyTorch is clearest, and it is optional unless your JAX driver involves TPUs or large-scale training. Pair it with ARENA Chapters 2-4.

- [ ] **Distributed arrays and automatic parallelization**, **Explicit sharding**, and **Manual parallelism with shard\_map** ([JAX docs](https://vfdev-5-jax.readthedocs.io/en/latest/extensions.html)). Meshes, `PartitionSpec`, how `jit` propagates shardings, and when to drop to `shard_map`.
- [ ] **Profiling computation** and **Profiling device memory** ([guides index](https://docs.jax.dev/en/latest/export/index.html)).
- [ ] **UvA "Training Models at Scale" (JAX)** ([UvA](https://uvadlc-notebooks.readthedocs.io/)): data, tensor and pipeline parallelism, ending in 3D-parallel language modelling.
- [ ] **"How to Scale Your Model"** by Google DeepMind ([book](https://jax-ml.github.io/scaling-book)):
  - TPU rooflines and compute- vs communication-bound regimes
  - sharded matmuls and transformer FLOP and memory accounting
  - training and inference parallelism, worked on LLaMA-3
  - closing chapters on profiling and programming TPUs in JAX

Compute: no multi-device hardware is needed to learn the APIs. JAX can expose several fake CPU devices for practice, and Colab offers TPUs for real runs.

Exit test:

- Shard your Stage 4 transformer's training step over a 2D mesh (data x model), and confirm parity with the single-device run.
- For one Scaling Book worked problem, compute the roofline estimate by hand before reading the answer.

Checkpoint question: when a sharded matmul becomes communication-bound, what quantity in the roofline model has crossed what threshold?

## Pairing schedule with ARENA

Each ARENA set you finish becomes the translation target for the next JAX stage. JAX stays one step behind ARENA, so you never learn a concept in both frameworks at once.

| After finishing ARENA | Do JAX stage | Translation target |
| --- | --- | --- |
| 0.1 Ray Tracing | Stage 1 | Ray tracer in `jax.numpy` |
| 0.2 CNNs & ResNets | Stage 2 (part A) | Pure-function MLP/conv, `grad`, per-example gradients |
| 0.3 Optimization | Stage 2 (part B) | SGD, RMSprop, Adam as pure functions |
| 0.4 Backprop | Stage 3 | Autograd vs `jax.grad`; training loop as `lax.scan` |
| 1.1 Transformers from Scratch | Stage 4 | Transformer in Flax NNX + Optax, logit parity |
| 1.2 Intro to Mech Interp | Stage 4 (Penzai) | One induction-head analysis in Penzai |
| Chapter 1 branches | Stage 5 | jaxpr interpreter as activation cache |
| Chapters 2-4 | Stage 6 (optional) | Sharded training step for the Stage 4 transformer |

Rule: if a JAX stage starts delaying the next ARENA set, pause JAX, not ARENA. ARENA is the primary track.

## Sources

- [JAX, Quickstart: How to think in JAX](https://melissawm.github.io/jax/notebooks/thinking_in_jax.html)
- [JAX, The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JAX 101 tutorial index](https://docs.jax.dev/en/latest/_sources/jax-101.rst)
- [JAX advanced guides index](https://docs.jax.dev/en/latest/export/index.html) and [sharding guides](https://vfdev-5-jax.readthedocs.io/en/latest/extensions.html)
- [JAX 601: internals, Autodidax](https://docs.jax.dev/en/latest/601/index.html)
- [Flax, Linen to NNX guide](https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html) and [Flax docs index](https://flax.readthedocs.io/en/stable/_sources/index.rst)
- [Optax, Getting started](https://optax.readthedocs.io/en/latest/getting_started.html)
- [Kidger & Garcia, Equinox (2021)](https://arxiv.org/abs/2111.00254)
- [Penzai docs](https://penzai.readthedocs.io/en/latest/_sources/index.rst) and [Johnson, Penzai + Treescope (ICML 2024 MI workshop)](https://arxiv.org/html/2408.00211v1)
- [UvA Deep Learning Tutorials, JAX track](https://uvadlc-notebooks.readthedocs.io/)
- [Google DeepMind, How to Scale Your Model](https://jax-ml.github.io/scaling-book)
- [DeepLearning.AI, Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax)
- [Eric Ma, Differential computing with JAX](https://ericmjl.github.io/notes/differential-computing-with-jax)
- [Apple, Accelerated JAX on Mac](https://developer.apple.com/metal/jax/), [Apple Developer Forums thread](https://origin-devforums.apple.com/forums/thread/815139), [metaljax on PyPI](https://pypi.org/project/metaljax/)
