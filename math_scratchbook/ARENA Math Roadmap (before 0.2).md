# ARENA Math Roadmap (before 0.2)

Nov 4, 2026 · @Adrino

## Scope

This roadmap covers two layers of math. The first is ARENA's own stated prerequisites, all completed before 0.2. The second is the math later chapters assume without teaching, which is deferred to just before each chapter needs it. Modules are gated by a diagnostic, not by dates: you already report solid linear algebra and probability, so most of the time should go to whatever the diagnostic shows is missing.

| Area | ARENA's stated requirement ([0.0 summary](https://deepwiki.com/theogbrand/arena_3.0/3.1-prerequisites)) | What later chapters use beyond it |
| --- | --- | --- |
| Neural networks | Structure, gradient descent, backprop, why nonlinearity, ReLU vs sigmoid | none extra |
| Linear algebra | Linear maps, rank, trace, determinant, transpose, bases and change of basis, matmul; SVD and eigen optional | SVD and low-rank factorisation for QK/OV circuits (1.2); near-orthogonality in high dimensions (1.5.4, SAEs) |
| Calculus | Derivatives, partials, basic vector calculus, chain rule, gradient descent | Jacobians and vector-Jacobian products for your own autograd (0.4) |
| Probability & stats | Probability rules, expectation, standard deviation, independence, normal distribution, combining random variables | Sampling and temperature (1.1); estimators, variance and baselines (Chapter 2); ELBO (0.5, optional) |
| Information theory | Entropy, KL, cross-entropy; marked helpful, not essential | KL penalties in RLHF (2.4); log-probs throughout |

Why the linear algebra column is the longest: in the transformer circuits framework, each head's QK and OV matrices are nominally vocab-by-vocab but only rank `d_head` (64 or 128). They are read as low-rank factorisations ([review of Elhage et al.](https://induction1.github.io/notes/transformer-circuits/index.html)). Rank, factorisation and SVD are working tools in ARENA 1.2, not background.

## Diagnostic (do this first)

Take this closed-book in one sitting, by hand. If a module's problems are fully correct with derivations, skip that module's learning steps and do only its exit test. Answers are deliberately not given; check them by deriving a second way or by running code.

**Linear algebra (Module 1)**

1. `W_Q, W_K` are `(d_model, d_head)` with `d_head < d_model`. What is the maximum rank of `W_Q W_K^T`, and why?
2. State the SVD of a real `m x n` matrix. Then say what the top singular vector pair of `W_OV` tells you about a head.
3. Why is the trace of `AB` equal to that of `BA` even when `AB` and `BA` have different shapes?
4. Under what condition is a change of basis an isometry, and why does that matter for whether neuron directions are "privileged"?

**Calculus (Module 2)**

5. Derive the gradient of `L = ||Wx - y||^2` with respect to `W`, with shapes.
6. Derive `dL/dz` for `L = -log softmax(z)[y]`, as a vector.
7. For `f: R^n -> R^m`, what is the shape of the Jacobian, and why does reverse-mode autodiff never build it explicitly?

**Probability (Module 3)**

8. `X, Y` are independent with variances `a, b`. What is `Var(2X - 3Y)`?
9. Explain why the sample mean's standard error shrinks as `1/sqrt(n)`.
10. Write `E[f(X)]` as a Monte Carlo estimate, and state when it is unbiased.

**Information theory (Module 4)**

11. Show `H(P, Q) = H(P) + KL(P || Q)`, and say why minimising cross-entropy loss is the same as minimising KL to the data distribution.
12. Give a two-outcome example where `KL(P||Q) != KL(Q||P)`.

**Optimization (Module 5)**

13. Why does too large a learning rate diverge on a quadratic? Express the threshold in terms of the curvature.
14. What problem does Adam's second-moment estimate solve that momentum does not?

## Module 1: Linear algebra

This is the most-used math in ARENA, and the module to over-invest in. Every concept below is verified in code, not only on paper.

- [ ] **Geometric intuition: 3Blue1Brown, Essence of Linear Algebra** ([playlist via curated list](https://awesome.ecosyste.ms/projects/github.com%2FLearnMachineLearning%2Fthe-math-of-intelligence)). Watch at 1.5x as a re-grounding: linear maps, change of basis, determinant as volume, eigenvectors.
- [ ] **Formal core: *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong), Chapters 2-4.** Linear algebra, analytic geometry (inner products, projections, orthogonality) and matrix decompositions (eigen, SVD). The PDF is free from the authors ([mml-book.com](https://mml-book.com), [table of contents](https://aimath.org/?p=8172)). Do the chapter exercises; solutions are available.
- [ ] **Depth, optional: Strang, MIT 18.065, *Matrix Methods in Data Analysis, Signal Processing, and Machine Learning*** ([course listing](https://awesome.ecosyste.ms/projects/github.com%2Fmachine-learning-tokyo%2Fmath_resources)). Take only the SVD and low-rank approximation lectures.
- [ ] **High-dimensional geometry: Princeton COS 521 lecture notes on high-dimensional geometry** ([notes](https://www.cs.princeton.edu/courses/archive/fall16/cos521/Lectures/lec9.pdf)). In `R^d` you can fit about `exp(cd)` unit vectors with all pairwise angles near 90 degrees, plus the Johnson-Lindenstrauss lemma. This is the geometric basis of superposition ([explainer](https://ttsugriy.github.io/mechinterp-first-principles/chapters/06-superposition.html)).

Code checks, in PyTorch:

- Compute the SVD of a random `(64, 512)` matrix. Reconstruct it, then truncate to rank `k` and plot error vs `k`.
- Sample 10,000 random unit vectors in `d = 768` and plot the histogram of pairwise cosine similarities.
- Form `W_Q W_K^T` from random `(512, 64)` factors and confirm its rank numerically.

Exit test: redo diagnostic items 1-4 cold. Then explain, in a paragraph with no notation lookups, why a rank-64 matrix can still be "read" as a 50k x 50k token-to-token table.

Checkpoint question: the cosine histogram narrows as `d` grows. What does that predict about how many features a model can store with tolerable interference, and what does sparsity add?

## Module 2: Multivariable and matrix calculus

The target is deriving gradients through matrix expressions, with shapes, fast enough that ARENA 0.4 (your own backprop) is implementation work, not derivation work.

- [ ] **3Blue1Brown, Neural Networks, chapters on gradient descent and backpropagation**, ending with "Backpropagation calculus" ([series](https://ki-campus.org/node/209)). It shows how ML practitioners think of the chain rule over a network, which differs from a typical calculus course ([summary](https://rosetta.to/u/3blue1brown/backpropagation-calculus-deep-learning-chapter-4)).
- [ ] **Parr & Howard, "The Matrix Calculus You Need For Deep Learning"** ([HTML](https://explained.ai/matrix-calculus/), [arXiv](https://arxiv.org/abs/1802.01528v1)). It assumes only first-year calculus and builds Jacobians, element-wise operations and the vector chain rules from first principles. Work every example on paper.
- [ ] **MML book, Chapter 5: Vector Calculus** ([mml-book.com](https://mml-book.com)): gradients of matrices, backprop as automatic differentiation, linearisation and Taylor series.
- [ ] **Vector-Jacobian products.** For linear, ReLU, softmax and layernorm, write the backward pass as a VJP `(upstream grad, saved inputs) -> input grad` without forming the Jacobian.

Code check: for each VJP you derived, compare against `torch.autograd.grad` with `torch.autograd.gradcheck` on float64 inputs.

Exit test: redo diagnostic items 5-7 cold. Then derive, in one sitting, the full backward pass of a 2-layer MLP with softmax cross-entropy, with shapes, and match it to autograd.

Checkpoint question: softmax's Jacobian is `diag(p) - p p^T`. Why does its VJP never need the `n x n` matrix, and what does that save at vocabulary size 50k?

## Module 3: Probability and statistics

ARENA needs probability as a working language: expectations, variance and estimators, and likelihoods. Measure theory and combinatorics are not needed. Select lectures rather than taking a full course.

- [ ] **Harvard Stat 110 (Blitzstein): free lectures and free second-edition textbook** ([course site](https://stat110.hsites.harvard.edu/), [book availability](https://datasciencecentral.com/?p=48501)). Take these blocks of the 34 lectures ([lecture map](https://stat110-notes.pages.dev/)):
  - Conditional probability, lectures 4-6
  - Random variables and expectation, 7-11
  - Continuous random variables, 12-16
  - Joint distributions, 17-22
  - Conditional expectation, 26-28
  - Limit theorems (LLN, CLT), 29-34 Skip counting (1-3) and order statistics unless the diagnostic flagged them.
- [ ] **MML book, Chapter 6: Probability and Distributions**, plus **Section 8.3, Parameter Estimation**, for maximum likelihood and MAP ([table of contents](https://www.cs.columbia.edu/~verma/classes/ml/ref/ref_math_for_ml.pdf)).
- [ ] **Estimators.** Bias, variance and mean squared error. Monte Carlo estimates of expectations, and why subtracting a baseline lowers variance without adding bias. Chapter 2's policy-gradient sets rely on this last point.

Code checks:

- Simulate the CLT: means of `n` draws from an exponential distribution, and the histogram of their standard error as `n` grows.
- Estimate `E[X^2]` by Monte Carlo and show its error shrinking at the `1/sqrt(n)` rate on a log-log plot.
- Show that subtracting a constant baseline `b` from `f(X)` in a score-function estimator leaves the mean unchanged but changes the variance.
- Derive the MLE of a categorical distribution and confirm that it equals the empirical frequencies.

Exit test: redo diagnostic items 8-10 cold. Then show that minimising negative log-likelihood of a categorical model is cross-entropy minimisation.

Checkpoint question: sampling at temperature `T` divides logits by `T`. What happens to the distribution's entropy as `T -> 0` and `T -> infinity`, and why?

## Module 4: Information theory

ARENA marks this as helpful rather than essential. It is still worth a full pass: cross-entropy is the loss everywhere, and KL divergence is the leash in RLHF.

- [ ] **Olah, "Visual Information Theory"** ([colah.github.io](https://colah.github.io/posts/2015-09-Visual-Information/)): entropy as optimal code length, cross-entropy, KL and mutual information, built from pictures. Practitioners often cite it as the explanation that made the subject click ([example](https://rhezab.readthedocs.io/en/latest/notes/info_theory.html)).
- [ ] **Formal pass:** a short chapter treatment of entropy, conditional entropy, KL and mutual information, e.g. McGill COMP 451, Chapter 12 ([PDF](https://cs.mcgill.ca/~wlh/comp451/files/comp451_chap12.pdf)).
- [ ] **Derive from scratch:**
  - the identity `H(P,Q) = H(P) + KL(P||Q)` ([worked example](https://mtuann.github.io/math-for-research/topics/advanced/information-theory/concepts/entropy-cross-entropy-and-kl-divergence.html))
  - why `KL >= 0` (Jensen's inequality)
  - why `KL` is asymmetric, and what forward vs reverse KL each penalise

Code checks:

- Implement entropy, cross-entropy and KL for categorical distributions using `log_softmax` only. Match `torch.nn.functional.cross_entropy` and `kl_div`, noting `kl_div`'s argument order.
- Fit a single Gaussian to a bimodal mixture twice, once with forward KL and once with reverse KL. Plot both fits.

Exit test: redo diagnostic items 11-12 cold. Then explain in three sentences why an RLHF objective adds a KL penalty to a reference model, and which direction of KL it uses.

Checkpoint question: the bimodal experiment gives one fit that covers both modes and one that picks a single mode. Which KL produces which, and why does that follow from where each places its `log` weight?

## Module 5: Optimization

ARENA 0.3 has you build optimizers. The math needed is the local behaviour of gradient descent on curved loss surfaces; convex-optimisation theory beyond that is not needed.

- [ ] **MML book, Chapter 7: Continuous Optimization** ([mml-book.com](https://mml-book.com)): gradient descent, step size, momentum, stochastic gradients, and a brief look at convexity and constraints.
- [ ] **Goh, "Why Momentum Really Works" (Distill, 2017)** ([distill.pub](https://distill.pub/2017/momentum/)): gradient descent analysed on a quadratic via its eigenvalues, the condition number, and how momentum changes convergence. This links back to Module 1.
- [ ] **Kingma & Ba, "Adam"** ([arXiv](https://arxiv.org/abs/1412.6980)), Sections 1-3: first and second moment estimates, and bias correction.
- [ ] **Derive:** the maximum stable learning rate on `f(x) = x^T A x / 2`, in terms of the largest eigenvalue of `A`.

Code checks:

- On a 2-D ill-conditioned quadratic, plot the trajectories of SGD, momentum and Adam from the same start. Measure steps to tolerance as the condition number grows.
- Push the learning rate past your derived bound and confirm divergence at the predicted value.

Exit test: redo diagnostic items 13-14 cold. Then implement SGD with momentum and Adam from your derivation alone, before you start ARENA 0.3.

Checkpoint question: Adam divides by a per-parameter running RMS of gradients. In terms of the quadratic's eigenbasis, what is that an approximation of, and where does the approximation fail?

## Deferred math (just in time, not before 0.2)

These topics are needed later. Learning them now would delay 0.2 for material you would forget before using, so each is listed with its trigger section.

| Before ARENA | Math | Source |
| --- | --- | --- |
| 0.5 VAEs & GANs (optional) | ELBO, reparameterisation trick, minimax objectives | MML book Ch. 6 and 8.4 ([TOC](https://www.cs.columbia.edu/~verma/classes/ml/ref/ref_math_for_ml.pdf)) |
| 1.2 Intro to Mech Interp | Residual stream as a linear space; QK and OV circuits as low-rank bilinear and linear maps; virtual weights; composition | Elhage et al., "A Mathematical Framework for Transformer Circuits" ([review](https://induction1.github.io/notes/transformer-circuits/index.html)) |
| 1.5.4 Superposition, 1.3.3 SAEs | Almost-orthogonal packing, Johnson-Lindenstrauss, sparsity and L1 penalties, phase changes | Elhage et al., "Toy Models of Superposition" ([walkthrough](https://deep-paper.org/en/paper/2209.10652)); Module 1 geometry notes |
| 1.5.2 Grokking & Modular Arithmetic | Discrete Fourier transform, modular arithmetic, trig identities | Introduced inside the set; review the DFT first |
| Chapter 2 RL | MDPs, Bellman equations, policy gradient theorem, advantage estimation, importance ratios | Sutton & Barto, cited by ARENA ([PDF](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)); OpenAI Spinning Up, in ARENA's reading list |
| Chapter 3 Evals | Binomial standard errors, confidence intervals, multiple-comparison caution | Stat 110 limit-theorem lectures (Module 3) |

Rule: when you reach a trigger section, spend at most one session on its math, then start the exercises and learn the rest from the code.

## Order and gate before 0.2

Run the modules in the order 1, 2, 3, 4, 5. Modules 3 and 4 can be interleaved, since entropy needs only expectation. This roadmap can overlap the PyTorch roadmap, because its code checks double as PyTorch practice. Start 0.2 only when the gate below passes, unaided.

- [ ] All 14 diagnostic items correct, cold, with derivations
- [ ] Every module's exit test passed
- [ ] **Synthesis problem:** for a one-hidden-layer ReLU network with softmax output, by hand:
  - write the loss as cross-entropy and as a KL plus a constant
  - derive all gradients with shapes
  - state the maximum stable SGD learning rate for its linearised loss near a minimum
  - code it in PyTorch and confirm all three against autograd and an actual training run

If the synthesis problem fails on one part, return to that module only.

Checkpoint question: which of the five modules did the diagnostic say you could skip, and what evidence would show that judgement was wrong once you are inside ARENA?

## Sources

- [ARENA 0.0 Prerequisites, DeepWiki summary](https://deepwiki.com/theogbrand/arena_3.0/3.1-prerequisites)
- [Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*](https://mml-book.com) ([TOC](https://aimath.org/?p=8172), [full PDF mirror at Columbia](https://www.cs.columbia.edu/~verma/classes/ml/ref/ref_math_for_ml.pdf))
- [3Blue1Brown, Essence of Linear Algebra](https://awesome.ecosyste.ms/projects/github.com%2FLearnMachineLearning%2Fthe-math-of-intelligence) and [Neural Networks series](https://ki-campus.org/node/209)
- [Strang, MIT 18.065](https://awesome.ecosyste.ms/projects/github.com%2Fmachine-learning-tokyo%2Fmath_resources)
- [Princeton COS 521, High-dimensional geometry notes](https://www.cs.princeton.edu/courses/archive/fall16/cos521/Lectures/lec9.pdf)
- [Parr & Howard, The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
- [Harvard Stat 110](https://stat110.hsites.harvard.edu/) and [lecture map](https://stat110-notes.pages.dev/)
- [Olah, Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
- [McGill COMP 451, Ch. 12 Information Theory](https://cs.mcgill.ca/~wlh/comp451/files/comp451_chap12.pdf)
- [Goh, Why Momentum Really Works](https://distill.pub/2017/momentum/)
- [Kingma & Ba, Adam](https://arxiv.org/abs/1412.6980)
- [Review of Elhage et al., A Mathematical Framework for Transformer Circuits](https://induction1.github.io/notes/transformer-circuits/index.html)
- [Toy Models of Superposition walkthrough](https://deep-paper.org/en/paper/2209.10652) and [superposition geometry chapter](https://ttsugriy.github.io/mechinterp-first-principles/chapters/06-superposition.html)
- [Sutton & Barto, Reinforcement Learning](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
