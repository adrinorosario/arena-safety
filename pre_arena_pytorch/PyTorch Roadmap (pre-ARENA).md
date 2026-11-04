# PyTorch Roadmap (pre-ARENA)

Nov 4, 2026 · @Adrino

## Target and assumptions

The target is ARENA's own library prerequisite list, met without documentation lookups. That list covers NumPy broadcasting, `torch.Tensor`, `nn.Parameter` and `nn.Module`, `.backward()`, losses and optimizers, and einops/einsum ([ARENA 0.0 summary](https://deepwiki.com/theogbrand/arena_3.0/3.1-prerequisites)). Stages are ordered by dependency and gated by exit tests, not dates, because weekly hours are not set.

| Assumption | If wrong |
| --- | --- |
| PyTorch only; JAX is out of scope here | A JAX track would follow Stage 4 |
| Math (linear algebra, probability) is already solid | A math refresher stage would precede Stage 3 |
| You have run training loops before (nanoGPT, TRL) but through high-level code | Stage 4 shrinks if you can already write a loop unaided |
| Compute is the M2 Air plus free Colab | Nothing here needs a GPU |

- [ ] Open question: how many hours a week can this take alongside coursework?

## Working method

Fluency here means predicting every tensor's shape before running the line. The steps below apply in every stage.

1. **Shape contract first.** Before coding, write input and output shapes with named dims, e.g. `(batch, seq, d_model) -> (batch, seq)`.
2. **Primitives before library calls.** Try indexing, broadcasting, `arange`, `where` and reductions first. Tensor Puzzles exists because learners reach for a library function when the base operations already suffice ([srush](https://github.com/srush/Tensor-Puzzles)).
3. **Assert, don't guess.** Put `assert t.shape == (...)` after each line until the contract holds.
4. **Lookup ledger.** Log every lookup: the op, what it does to shape, one example. An op you look up twice goes on a one-page op card that you review before each session.
5. **Re-derive cold.** For each exit test, rewrite the solution from a blank file the next day.

Checkpoint question: why does writing the shape contract *before* the code reduce lookups more than reading docs after an error?

## Stage 1: tensors, indexing, broadcasting

This stage is the root of most ARENA stalls, so it gets the most repetition. Start it today.

- [ ] **UvA Tutorial 2, Introduction to PyTorch** ([UvA DL notebooks](https://uvadlc-notebooks.readthedocs.io/)): tensor creation, dtype, device, `view`/`reshape`, `permute`, indexing. Runs on a laptop CPU.
- [ ] **PyTorch broadcasting semantics note** ([docs](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)). Learn the two rules: align trailing dims, each pair must be equal, 1, or missing; then prepend 1s and take the max per dim. Also note that broadcasting expands without copying, and that in-place ops cannot change the left operand's shape.
- [ ] **Tensor Puzzles, all 21** ([repo](https://github.com/srush/Tensor-Puzzles)). Each is one line under 80 columns, with only `arange`, `where`, indexing, broadcasting and arithmetic allowed. Watch Rush's walkthrough only after each attempt.

Exit test, no lookups:

- Predict `(x + y).shape` for `x: (5, 1, 4, 1)`, `y: (3, 1, 1)` before running it.
- Write `outer(a, b)`, `diag(A)`, `eye(n)` and a causal mask `(n, n)` using only `arange` and broadcasting.
- Gather one element per row from `(batch, classes)` given labels `(batch,)`, two ways: integer indexing and `gather`.

Checkpoint question: `torch.ones(4, 1) + torch.ones(4)` returns shape `(4, 4)`, not `(4,)`. Which rule produced that, and how would that silently corrupt a loss calculation?

## Stage 2: einops and einsum

ARENA uses einops and einsum throughout Chapter 1. An ARENA-derived GPT-2 workshop tells participants to prioritise these exercises above all its other prerequisites ([Build GPT-2](https://lu.ma/build-gpt2)).

- [ ] **einops tutorial, Part 1: Basics** ([official notebook](https://nbviewer.jupyter.org/github/arogozhnikov/einops/blob/main/docs/1-einops-basics.ipynb)). Covers `rearrange`, `reduce` and `repeat`, plus axis composition `(h h2)` and decomposition. The author shows these three ops cover reshape, transpose, squeeze, stack, tile and pooling ([einops](https://github.com/tqlong/einops-tutorial)).
- [ ] **"Einsum Is All You Need"**, recommended by the same workshop for the einsum convention.
- [ ] **Rewrite Stage 1 in einops.** Redo three Tensor Puzzles solutions with `einops` or `torch.einsum`, then compare readability.

Exit test, no lookups:

- `(b, c, h, w) -> (b, h, w, c)`, then flatten to `(b, h*w*c)`, in `rearrange`
- 2x2 max pooling on `(b, c, h, w)` in one `reduce`
- matmul, trace, batched outer product, and attention scores `(b, q, d), (b, k, d) -> (b, q, k)` in `einsum`
- split `d_model` into heads: `(b, s, h*d) -> (b, h, s, d)`

Checkpoint question: `x.view(x.shape[0], -1)` and `rearrange(x, 'b c h w -> b (c h w)')` give identical results. What does the second tell a reader, or a future you, that the first does not?

## Stage 3: autograd from the inside

Build autograd once by hand, then read PyTorch's own description of it. That pairing turns `.backward()` from a ritual into something you can predict.

- [ ] **Karpathy, Lecture 1: building micrograd** ([course](https://karpathy.ai/zero-to-hero.html)). A scalar autograd engine over a dynamically built graph, with topological sort and a reverse pass. The lecture checks its gradients against PyTorch ([notes](https://benzales.github.io/notes/zero-to-hero-arc.html)).
- [ ] **Karpathy, makemore lectures** in the same series. They move from scalars to PyTorch's tensor autograd across bigram, MLP and later models ([makemore notes](https://benzales.github.io/notes/repos/makemore.html)). Prioritise the one where gradients are computed by hand through every tensor op.
- [ ] **PyTorch Autograd mechanics note** ([source](https://pytorch.org/docs/2.3/_sources/notes/autograd.rst.txt)):
  - only leaf tensors with `requires_grad=True` accumulate `.grad`
  - non-leaf tensors need `retain_grad()` to keep their gradient
  - grad mode, no-grad mode and inference mode
  - saved tensors

Exit test, no lookups:

- Derive and code the gradient of softmax cross-entropy with respect to the logits; match `torch.autograd` to `1e-6`.
- Predict which of `w`, `h = w @ x`, `loss` have `.grad` after `loss.backward()`, and why.
- Show what goes wrong if `optimizer.zero_grad()` is omitted, using a two-step example.

Checkpoint question: gradients *accumulate* into `.grad` instead of overwriting it. What use case does that design serve, and what bug does it invite?

## Stage 4: nn.Module, optimizers, training loops

The goal of this stage is to know what each PyTorch abstraction replaces, by first writing the code it replaces.

- [ ] **"What is torch.nn really?"** by Jeremy Howard, in the official tutorials ([PyTorch](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)). It trains MNIST with raw tensors first. Then it adds `torch.nn.functional`, `nn.Module`/`nn.Parameter`, `torch.optim`, `Dataset` and `DataLoader` one at a time, showing what each replaces. Type every step; do not run the downloaded notebook.
- [ ] **UvA Tutorial 2, second half** ([UvA](https://uvadlc-notebooks.readthedocs.io/)): `nn.Module`, data loading, a full train/eval loop.
- [ ] **Write SGD and Adam by hand** as plain functions that update `p.data` under `torch.no_grad()`. Then match `torch.optim` step for step on a small model.

Exit test, no lookups, blank file:

- An MLP on MNIST: `Dataset`, `DataLoader`, `nn.Module`, loss, optimizer, train loop, eval loop with `model.eval()` and `torch.no_grad()`.
- Count trainable parameters, freeze one layer, and confirm its weights did not change after a step.
- Save and reload a `state_dict`; resume training and confirm the loss continues from the same value.

Checkpoint question: what exactly does `model.eval()` change, and why is it separate from `torch.no_grad()`?

## Stage 5: debugging and research workflow

Research code fails silently more often than loudly, so this stage drills detection.

- [ ] **UvA Guide 3: Debugging in PyTorch** and **Guide 2: Research projects with PyTorch** ([UvA guides](https://uvadlc-notebooks.readthedocs.io/en/latest/search.html)).
- [ ] **Full Stack Deep Learning, "Debug" lecture** ([notes and video](https://fall2019.fullstackdeeplearning.com/course-content/training-and-debugging/debug)). It names incorrect tensor shapes among the five most common bugs. It also prescribes overfitting a single batch, reading rising error as a likely sign flip in the loss or gradient.
- [ ] **Forward hooks.** Use `register_forward_hook` to capture one layer's activations from your Stage 4 MLP, then remove the hook. ARENA's interpretability tooling is built on the same idea.
- [ ] **Bug hunt.** Plant one bug of each kind in your Stage 4 script and find each from symptoms alone: shape broadcast error, missing `zero_grad`, `eval()` left on, wrong loss input (probabilities instead of logits).

Exit test: the Stage 4 model overfits one batch to near-zero loss, and you can name the symptom each planted bug produced.

Checkpoint question: why is a model that *cannot* overfit a single batch evidence of a code bug, rather than a data or hyperparameter problem?

## Readiness gate before ARENA

Start ARENA 0.0 only when all four items below pass in one sitting, unaided. Any failure sends you back to that stage, not forward.

- [ ] **Tensors.** Implement `softmax`, `log_softmax`, `logsumexp` (numerically stable) and cross-entropy for `(batch, classes)` from primitives.
- [ ] **einops.** Implement multi-head attention scores and a causal mask for `(batch, seq, d_model)` input with `n_heads`, using `rearrange` and `einsum`.
- [ ] **Autograd.** Hand-derive one layer's backward pass and match `torch.autograd`.
- [ ] **Training.** Write an MLP training loop from a blank file, overfit one batch, then train on MNIST to above 95% validation accuracy.

If the gate passes, the 0.0 einops and einsum exercises should take minutes, not hours.

## Sources

- [ARENA 0.0 Prerequisites, DeepWiki summary](https://deepwiki.com/theogbrand/arena_3.0/3.1-prerequisites)
- [PyTorch, Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch, Autograd mechanics](https://pytorch.org/docs/2.3/_sources/notes/autograd.rst.txt)
- [Jeremy Howard, What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
- [UvA Deep Learning Tutorials (Lippe)](https://uvadlc-notebooks.readthedocs.io/)
- [Sasha Rush, Tensor Puzzles](https://github.com/srush/Tensor-Puzzles)
- [einops Basics tutorial](https://nbviewer.jupyter.org/github/arogozhnikov/einops/blob/main/docs/1-einops-basics.ipynb)
- [Build Your Own GPT-2 workshop prerequisites](https://lu.ma/build-gpt2)
- [Karpathy, Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), with [lecture notes](https://benzales.github.io/notes/zero-to-hero-arc.html) and [makemore notes](https://benzales.github.io/notes/repos/makemore.html)
- [Full Stack Deep Learning, Debug](https://fall2019.fullstackdeeplearning.com/course-content/training-and-debugging/debug)
