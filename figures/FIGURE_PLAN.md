# Figure plan

One folder per figure under `figures/src/<name>/<name>.tex`, built to
`figures/build/<name>.svg` by `bash figures/src/build_figures.sh`.

Palette every figure must use:

```latex
\definecolor{c1}{HTML}{000000}  % black       - queries, reference times, structure, axes
\definecolor{c2}{HTML}{233D4D}  % deep slate  - keys, observation times, the operative step
\definecolor{c3}{HTML}{FE7F2D}  % orange      - values, the measurements themselves
\definecolor{c4}{HTML}{EAECF0}  % cool grey   - panel fills, backgrounds
\definecolor{axiscolor}{HTML}{333333}
\definecolor{labelcolor}{HTML}{5F6B75}
```

Role convention held across the whole tutorial, so the reader learns the colours
once: **query time = c1 black, key / observation time = c2 slate, measured value
= c3 orange, background panel = c4**. Never swap these between figures.

A time axis is always horizontal, always left to right, always labelled in
hours, and observations are always drawn as ticks on it.

---

## Part I, sequences that are signals

### 01 From Tokens to Timestamps

| name | shows |
|------|-------|
| `sentence_vs_signal` | The five-word sentence from the first tutorial above, the patient's three channels below, both drawn as a row of boxes. The sentence boxes are indexed **(1)** to **(5)**; the signal boxes carry a clock. What survives the move and what does not. |

### 02 One Number Is Not a Word

| name | shows |
|------|-------|
| `token_information` | One embedding row for *mouse* against one scalar heart-rate reading of 78. The word carries a whole vector of meaning, the reading carries a number that means nothing without its neighbours. |
| `quadratic_wall` | Sequence length against attention cost, with a sentence (n = 5), a paragraph (n = 500), and 48 hours of minute-sampled vitals (n = 2880) marked on it. |

### 03 Patches

| name | shows |
|------|-------|
| `patching` | A strip of signal cut into overlapping windows of length $P$ at stride $S$, each window flattened and projected into one token. Shapes annotated at every step. |

### 04 Many Channels, One Clock

| name | shows |
|------|-------|
| `channel_mixing` | Two arrangements side by side. Left, channels concatenated into one token per timestep so attention mixes them. Right, each channel run through the same attention on its own. What each one can and cannot express. |

### 05 Forecasting Under a Causal Mask

| name | shows |
|------|-------|
| `horizon_mask` | The lookback window and the horizon on one axis, with the mask that separates them. Contrasts one-step autoregression with a direct multi-step head. |

## Part II, when the clock is not a ruler

### 06 Position Is Only a Proxy for Time

| name | shows |
|------|-------|
| `index_vs_time` | The same six observations twice. Above, plotted against index, evenly spaced. Below, plotted against the real timestamp, bunched and gapped. The positional encoding sees only the top row. |

### 07 The Irregular Series

| name | shows |
|------|-------|
| `three_problems` | The patient's chart as three tick strips on a shared 48-hour axis: heart rate dense, temperature occasional, lactate four ticks. Annotates the three distinct failures, uneven gaps, missing readings, and unaligned channels. The figure the whole of Part III answers. |

### 08 The Repairs That Do Not Hold

| name | shows |
|------|-------|
| `binning_damage` | The same tick strip discretised into hour-long bins: one bin holds three readings and has to pick, four bins hold none and invent. |
| `forward_fill` | A forward-filled temperature trace against the true one, with the flat plateau where nothing was measured, and the model unable to tell a plateau from a real steady reading. |

### 09 Embedding Continuous Time

| name | shows |
|------|-------|
| `pe_to_phi` | The sinusoidal encoding evaluated only at integer positions, then the same waves evaluated at arbitrary real $t$, with the observation times of the running example falling between the integers. |
| `time_embedding_terms` | The linear term and three sine terms of $\phi_h(t)$ drawn separately over 48 hours, and the vector they stack into at one chosen $t$. |

## Part III, multi-time attention

### 10 Attention as Interpolation

| name | shows |
|------|-------|
| `time_only_scores` | The standard attention score, content against content, beside the multi-time score, time embedding against time embedding. The value track is drawn in orange in both, and only in the second is it absent from the score. |
| `learned_kernel` | $\kappa(t, t')$ as a curve over $t'$ for one fixed query time $t$, next to a fixed RBF bump of the same width. The learned kernel is allowed to be asymmetric and multi-modal. |

### 11 Queries You Choose

| name | shows |
|------|-------|
| `reference_points` | Observation ticks along the bottom in slate, $K$ evenly spaced reference points along the top in black, and weighted edges between them. A ragged input becomes a regular output of length $K$. |

### 12 Missingness Without Imputation

| name | shows |
|------|-------|
| `per_channel_softmax` | One reference point querying three channels. Three separate softmaxes, each normalised over only the ticks that channel actually has. No imputed values anywhere in the picture. |
| `mask_channel` | The value track and the mask track side by side, and the two outputs they produce: an interpolated reading, and how much evidence stood behind it. |

### 13 The mTAN Network

| name | shows |
|------|-------|
| `mtan_encoder_decoder` | The full architecture. Encoder: observations as keys and values, reference points as queries, then the RNN and the per-reference-point latent. Decoder: the roles swapped, reference points as keys, arbitrary output times as queries. The one figure that shows the module used in both directions. |

### 15 What It Buys and What It Costs

| name | shows |
|------|-------|
| `attention_as_weights` | The learned attention weights drawn as edge thickness from reference points down to observations, the visual proof that the module is doing interpolation. |
