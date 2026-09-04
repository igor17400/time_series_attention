# mTAN — Multi-Time Attention Networks (Shukla & Marlin, ICLR 2021, arXiv 2101.10318)

Source: paper PDF (`mtan.pdf` / `mtan.txt`) + reference implementation (`mtan_models.py`, github.com/reml-lab/mTAN).

## 1. Problem

Sparse, irregularly sampled, **multivariate** time series (ICU records, human activity).
Two failures of standard nets:
- observation times are unevenly spaced (no fixed Δt),
- different **dimensions** are observed at *different* times (unaligned; a "vector" at time t is only partially observed).

Data case: `s_n = {s_dn}`, per-dimension tuple `s_dn = (t_dn, x_dn)` with its own length `L_dn`.
So the model never assumes a rectangular (T × D) matrix. Supervised task adds label `y_n`.

Prior fixes and their cost: binning (ad hoc, creates missingness), GRU-D / decay RNNs (can't handle partially observed vectors without imputation), IP-Nets (interpolation against reference points but with a **fixed RBF kernel**), latent ODEs (accurate but 1–2 orders of magnitude slower).

## 2. Core idea in one sentence

Re-represent the series at a **fixed set of K reference time points** by attention where
**queries and keys are embeddings of *time only*** and **values are the observed measurements**.
The attention weights are therefore a *learned similarity kernel over time* — a learnable replacement for the RBF kernel of IP-Nets.

Key contrast to a normal Transformer: content never enters the score. Attention logits are a function of `(t_query, t_key)` alone. Attention is used as **interpolation**, not as token mixing.

## 3. Time embedding (Eq. 1)

H embedding functions `φ_h : R → R^{d_r}`:

```
φ_h(t)[i] = ω_0h · t + α_0h            if i = 0     (linear term: trend / progression of time)
          = sin(ω_ih · t + α_ih)       if 0 < i < d_r  (periodic terms: periodicity)
```
`ω, α` **learnable**. = one-layer FC with sine nonlinearity mapping scalar t → R^{d_r}.
Generalises transformer positional encoding to continuous t (subsumes it when evaluated at discrete positions, and PE is fixed while this is learned).
Property: `φ_h(t+Δ)` is a linear function of `φ_h(t)` — so shifts are representable.

Ablation (Table 3): learned embedding beats fixed sinusoidal PE by ~1 AUC point (PhysioNet 0.858 vs 0.845; MIMIC 0.854 vs 0.843).

Code: `learn_time_embedding` = `cat([Linear(1,1)(t), sin(Linear(1, d_r-1)(t))], -1)`. d_r = 128 in experiments.

## 4. Multi-time attention module mTAN(t, s) (Eqs. 2–4)

Continuous function of query time `t`, returning a J-dim vector.

```
κ_h(t, t_id) = softmax_over_i [ φ_h(t) w vᵀ φ_h(t_id)ᵀ / sqrt(d_k) ]      (4)
x̂_hd(t, s)  = Σ_{i=1..L_d} κ_h(t, t_id) · x_id                            (3)
mTAN(t,s)[j] = Σ_{h=1..H} Σ_{d=1..D} x̂_hd(t,s) · U_hdj                    (2)
```

- `w`, `v` are `d_r × d_k` projection matrices → these ARE the query/key projections (`φ(t)w` = query, `φ(t')v` = key). Scale `1/sqrt(d_k)`, `d_k ≤ d_r`.
- **Softmax is taken per data dimension d**, over only the time points where d is actually observed. That is why no imputation step is needed: an unobserved dimension simply isn't in its own softmax.
- Eq. 3 = kernel smoother / interpolation on dimension d with a *learned* kernel κ.
- Eq. 2 = learned linear mix `U` (H × D × J) across heads and across data dimensions → mixes variables, gives compact J-dim output; sparsity in U lets different variables use different time embeddings.
- All query points computed in parallel (no recurrence, no ODE solve) → the speed win.

### Discretization → mTAND
`mTAND(r, s)[i] = mTAN(r_i, s)`, with reference points `r = [r_1..r_K]`.
Output: sequence of length |r|, each dim J → a regular, fixed-size sequence any RNN/CNN/FC can eat.
Reference points can be fixed, or `ρ(s)` = union of times observed on any dimension of s.
Classification uses K = 128 reference points; interpolation searches K ∈ {8,16,32,64,128}.

### What the reference code actually does (important deltas)
- Input tensor `x = [values ‖ mask]` → value dim is `2*D`. The mask channels get interpolated too, so the output carries a per-variable **local observation-density / intensity** channel alongside the interpolated value (same trick as IP-Nets).
- Scores are `(B, h, K, L)` from time embeddings, then `repeat_interleave` over the value dimension, `masked_fill(mask==0, -1e9)`, `softmax(dim=-2)` (over keys) → per-dimension normalisation as in Eq. 4.
- The H "embedding functions" are implemented as **multi-head split** of one shared d_r embedding: `Linear(d_r,d_r)` for query and key, reshaped into h heads of size `d_r/h`. Heads concat → `Linear(D*h, J)` = the `U` tensor. H ∈ {1,2,4}.
- Fixed-embedding fallback uses sinusoids with time × 48 (hours) and base 10, not 10000.

## 5. Encoder–decoder (mTAND-Full), a VAE

Latent: `z = [z_1..z_K]`, one latent vector **per reference point** (temporally distributed latent, not a single z_0 as in latent-ODE). This is what lets it capture local structure.

**Encoder** `q_γ(z | r, s)` (Eqs. 9–11):
1. `h_TAN = mTAND_enc(r, s)` — query = K reference points, key = observed times, value = observed values(+mask).
2. `h_RNN = biGRU(h_TAN)`.
3. per reference point k: `μ_k = f_μ(h_k)`, `σ²_k = exp(f_σ(h_k))`, `z_k ~ N(μ_k, σ²_k)`. (2-layer FC, 50 hidden, ReLU.)

**Decoder** `p_θ(x | z, t)` (Eqs. 5–8) — inverted, roles of query/key swap:
1. `z_k ~ p(z_k) = N(0, I)` (prior), IID over k.
2. `h_RNN = RNN_dec(z)` (biGRU) — gives the latent sequence temporal dependence.
3. `h_TAN = mTAND_dec(t, h_RNN)` — **query = the T output/observed times, key = the K reference points, value = decoder RNN hidden states**. No mask needed (all reference points valid).
4. `x_id ~ N(f_dec(h_i,TAN)[d], σ² I)`, fixed σ² = 0.01.

So mTAND is used twice, in both directions: irregular → regular (encoder), regular → arbitrary query times (decoder). The decoder can be queried at *any* t ⇒ interpolation/extrapolation for free.

## 6. Training

Normalised ELBO (Eq. 12): each data case's reconstruction+KL divided by its total number of observations `Σ_d L_dn`, so long sequences don't dominate.
- KL sums over the K reference points (Eq. 13).
- Log-lik sums only over *observed* (d, j) pairs (Eq. 14) — missingness handled here.
- k samples from q (5 for interpolation, 1 for classification). KL annealing 0.99 helped on PhysioNet.

Supervised (Eq. 15): `L = L_NVAE + λ · E_q[log p_δ(y|z)]`, classifier = GRU over z then 2-layer FC (300 units). λ = 100 (PhysioNet), 5 (MIMIC-III); Human Activity best with **no** ELBO term at all (pure discriminative).
Prediction: `y* = argmax_y E_q[log p_δ(y|z)]`.

**mTAND-Enc** ablation = mTAND module → GRU → FC classifier. No VAE. Often nearly as good and even faster.

## 7. Results (what to quote)

Datasets: PhysioNet 2012 (37 vars, 48h, 8000 series, in-hospital mortality, 13.8% positive), MIMIC-III (12 vars, 53,211 records, 8.1% positive), Human Activity (12 channels, 50 time points, per-time-point 11-class).

- Interpolation on PhysioNet (MSE ×10⁻³, 50–90% observed): mTAND-Full **4.14–4.80** vs L-ODE-ODE 6.72–7.14, L-ODE-RNN ~8.1, RNN-VAE 11–13. Large margin.
- Classification AUC: PhysioNet mTAND-Full **0.858**, mTAND-Enc 0.854, vs ODE-RNN 0.833, L-ODE-ODE 0.829, GRU-D 0.818. MIMIC-III mTAND-Full 0.8544 ≈ ODE-RNN 0.8561 (not significant). Human Activity accuracy mTAND-Full **0.910** vs ODE-RNN 0.885.
- Speed: **0.2 min/epoch** (mTAND-Full), 0.1 (mTAND-Enc) vs ODE-RNN 16.5 and L-ODE-ODE 22.0 → ~85–110× faster.
- Ablations: learned kernel ≥ fixed RBF (mTAND-Enc vs IP-Nets: 0.854/0.842/0.907 vs 0.819/0.839/0.869); learned time embedding > fixed PE.
- Synthetic (A.2): mTAN reconstructions track local structure; latent-ODE over-smooths. MSE 0.0028 vs 0.0191 (recon, latent dim 20).
- A.5 visualises attention weights as edges from reference points to observed points = literally a learned interpolation kernel.

## 8. Why this matters for QKV pedagogy

- Cleanest possible example of **decoupling where attention looks from what it retrieves**: queries/keys are pure *position* (continuous time), values are pure *content*. In a vanilla Transformer both come from the same token.
- Positional encoding, promoted from an additive hint to *the entire similarity signal*, and made continuous + learnable.
- Attention as **kernel regression / interpolation** — softmax weights are literally a normalised kernel `κ(t, t')`, Eq. 3 is Nadaraya–Watson with a learned kernel.
- Queries need not come from the data: they're a chosen grid of K reference points. Attention output length is set by the query set, not the input length ⇒ variable-length → fixed-length.
- Per-dimension masked softmax = handling missing data without imputation.
- Heads = several different notions of temporal similarity at once (different learned frequencies), later mixed by U.

## 9. Gaps / caveats

- Score is time-only, so two observations at the same time are weighted identically regardless of value — no content-based retrieval within the module (the RNN afterwards adds that).
- The mTAND module is not causal by itself: it attends to past and future observations (fine for interpolation/smoothing, needs masking for forecasting).
- Paper's H separate φ_h vs code's shared-embedding multi-head split is a real discrepancy.
