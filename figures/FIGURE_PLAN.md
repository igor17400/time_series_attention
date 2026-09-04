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

## 01 From Tokens to Timestamps

| name | shows |
|------|-------|
| `sentence_vs_signal` | **Drawn.** Two lanes into one pipeline. Top: the four tokens *the patient is stable* as boxes. Bottom: three channels on a 48-hour axis, ticked at every minute, with minute 300 pulled out as $[78, 37.2, 2.4]$. Each lane passes through a linear layer and a positional encoding into an $X$, four rows in one case and 2880 in the other, and both enter the same unchanged attention block. The visual argument that the mechanism needed no modification, only the input did. |

## 02 One Number Is Not a Word

| name | shows |
|------|-------|
| `token_information` | **Drawn.** Left: the token *stable* and the embedding it maps to, the same vector wherever it sits. Right: three pulse trajectories over the hour before minute 300, falling from 120, climbing from 60, and flat at 78, all meeting at 78 and collapsing onto the identical row $[78, 37.2, 2.4]$. Shows that the query and the key are built from an instant, so three clinically different patients are indistinguishable to the mechanism. |

`quadratic_wall` was dropped. A curve of $n^2$ restates the numbers already in the
prose (400 scores against eight million) and shows no mechanism.

## 03 Patches

| name | shows |
|------|-------|
| `patching_overlap` | **Drawn.** The `patching` figure rebuilt at $S = 30$, so the two can be read against each other. The windows are staggered rather than tiled because they overlap, the first three carry three different tints, and dashed guides carry each into the matching row of the $95 \times 60$ matrix and on into the token sequence. $N = 95$ throughout. |
| `stride_rows` | Superseded by `patching_overlap` and no longer used on the page. Source and SVG kept.

**Drawn.** Three consecutive $S = 30$ windows on a time axis (minutes 300 to 420), then the same three as rows 11, 12 and 13 of the $95 \times 60$ matrix. Each row is split into its two halves and labelled by minute range, and the halves that repeat between adjacent rows carry the same fill and are tied by dashed lines. Makes overlap concrete in the matrix, where the rows are aligned even though the windows are not. |
| `stride_overlap` | **Drawn.** Minutes 240 to 480 of the pulse, with a jump at 355 to 365 marked by a shaded band running the height of the figure. Above, $S = P = 60$: the windows tile and a seam lands at minute 360, splitting the jump. Below, $S = 30$: seven staggered bars, and the one covering minutes 330 to 389 is picked out because it holds the jump whole. The figure the stride paragraph needs, since the split is a thing to see rather than to be told. |
| `patching` | **Drawn.** Two rows. Above: the 48-hour signal with $L$ braced across it, cut into windows, with $P$ braced under one window and $S$ as a double arrow between two window starts. Below: the $N \times P$ matrix of patches, the learned $P \times d_{\text{model}}$ projection, the $N \times d_{\text{model}}$ token sequence, and attention. Dashed guides carry patch 1 down to row 1, and row 1 is tinted in both matrices, so the reader can trace one window all the way to one token. Every shape in the paragraph appears once in the figure. |

## 04 Many Channels, One Clock

| name | shows |
|------|-------|
| `channel_stacking` | **Drawn.** Heart rate, temperature and lactate on one clock, all three climbing during a highlighted hour. Dashed guides carry the three 60-minute windows down into one row of $D \times P = 180$ numbers laid end to end, then a learned $(D \cdot P) \times d_{\text{model}}$ matrix turns that row into one token. Placed at the case *for* stacking, so the reader sees the cross-channel event landing inside a single token. |
| `channel_fork` | **Drawn.** The two arrangements as two pipelines with identical stages, tokens then attention then outputs then head, so only the difference is visible. Left, stacking: one sequence, every token striped with all three channel colours, marked *the channels meet here* at the token row. Right, independence: three single-colour sequences through the same attention, marked *and here* at the head. Placed after the verdict paragraph, whose claim is that the fork is about where the channels meet. |
| `channel_mixing` | Superseded by `channel_fork`, not drawn. Left, channels concatenated into one token per timestep so attention mixes them. Right, each channel run through the same attention on its own. What each one can and cannot express. |

## 05 Forecasting Under a Causal Mask

| name | shows |
|------|-------|
| `causal_mask_grid` | **Drawn.** A $6 \times 6$ score table, rows labelled *the hour doing the reading* and columns *the hour being read*. Upper triangle filled with $-\infty$, lower triangle and diagonal filled. Row 3 outlined and annotated. Footnote states that $-\infty$ is added before the softmax. |
| `horizon_no_mask` | **Drawn.** Two panels. Left, one step at a time: six hours, an arrow from hour 3 to hour 4 labelled as what it must produce, and a crossed-out dashed arc from hour 3 to hour 6 as what it must not read. Right, the whole horizon: hours 1 to 48 with a double-headed arc reading either direction, into a head, out to hours 49 to 54 in orange. Carries the section's claim that the mask answers the recipe, not the task. |

## 06 Position Is Only a Proxy for Time

| name | shows |
|------|-------|
| `index_vs_time` | **Drawn.** Five windows twice over. Above, evenly spaced by index, every neighbouring pair joined by an arrow reading *one step*, the pair 2 to 3 in orange. Below, the same five placed on a minute axis at 0, 60, 400, 460 and 520, with the 2 to 3 gap braced as 340 minutes in orange and the others marked 60. Dashed guides join each token to itself. Shows the encoding asserting adjacency rather than merely omitting the gap. |

## 07 The Irregular Series

| name | shows |
|------|-------|
| `three_problems` | The patient's chart as three tick strips on a shared 48-hour axis: heart rate dense, temperature occasional, lactate four ticks. Annotates the three distinct failures, uneven gaps, missing readings, and unaligned channels. The figure sections 10 to 13 answer. |

## 08 The Repairs That Do Not Hold

| name | shows |
|------|-------|
| `binning_damage` | The same tick strip discretised into hour-long bins: one bin holds three readings and has to pick, four bins hold none and invent. |
| `forward_fill` | A forward-filled temperature trace against the true one, with the flat plateau where nothing was measured, and the model unable to tell a plateau from a real steady reading. |

## 09 Embedding Continuous Time

| name | shows |
|------|-------|
| `pe_to_phi` | The sinusoidal encoding evaluated only at integer positions, then the same waves evaluated at arbitrary real $t$, with the observation times of the running example falling between the integers. |
| `time_embedding_terms` | The linear term and three sine terms of $\phi_h(t)$ drawn separately over 48 hours, and the vector they stack into at one chosen $t$. |

## 10 Attention as Interpolation

| name | shows |
|------|-------|
| `time_only_scores` | The standard attention score, content against content, beside the multi-time score, time embedding against time embedding. The value track is drawn in orange in both, and only in the second is it absent from the score. |
| `learned_kernel` | $\kappa(t, t')$ as a curve over $t'$ for one fixed query time $t$, next to a fixed RBF bump of the same width. The learned kernel is allowed to be asymmetric and multi-modal. |

## 11 Queries You Choose

| name | shows |
|------|-------|
| `reference_points` | Observation ticks along the bottom in slate, $K$ evenly spaced reference points along the top in black, and weighted edges between them. A ragged input becomes a regular output of length $K$. |

## 12 Missingness Without Imputation

| name | shows |
|------|-------|
| `per_channel_softmax` | One reference point querying three channels. Three separate softmaxes, each normalised over only the ticks that channel actually has. No imputed values anywhere in the picture. |
| `mask_channel` | The value track and the mask track side by side, and the two outputs they produce: an interpolated reading, and how much evidence stood behind it. |

## 13 The mTAN Network

| name | shows |
|------|-------|
| `mtan_encoder_decoder` | The full architecture. Encoder: observations as keys and values, reference points as queries, then the RNN and the per-reference-point latent. Decoder: the roles swapped, reference points as keys, arbitrary output times as queries. The one figure that shows the module used in both directions. |

## 15 What It Buys and What It Costs

| name | shows |
|------|-------|
| `attention_as_weights` | The learned attention weights drawn as edge thickness from reference points down to observations, the visual proof that the module is doing interpolation. |
