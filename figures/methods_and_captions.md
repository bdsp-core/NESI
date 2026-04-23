# Methods and figure captions

## Methods

### Ordinal-logistic model of score-vs-NESI

We model each discrete clinical rating scale (GCS, RASS, CAMS, ICANS) as a function of the continuous Neural EEG Severity Index (NESI) using a proportional-odds cumulative-link (ordinal logistic) model. Let Y_t ∈ {0, 1, …, K−1} denote the scale's score in a direction-normalized encoding (higher = worse clinical state; GCS and RASS are reversed so that all scales share this convention). The model is

P(Y_t ≤ k | NESI) = σ(θ_k − β · NESI),   k = 0, 1, …, K−2

where σ(·) is the standard logistic function, θ_0 < θ_1 < … < θ_{K−2} are ordered category thresholds, and β is a single slope parameter shared across categories (the proportional-odds assumption). Parameters are obtained by numerical maximum-likelihood (L-BFGS-B) with θ re-parameterized as (θ_0, log(θ_1 − θ_0), log(θ_2 − θ_1), …) so the thresholds remain strictly ordered. Each scale is fit independently on the Train split of the NESI data set.

From the fit, per-category probabilities P(Y_t = k | NESI) are obtained by differencing the cumulative probabilities, and we derive conditional moments

E[Y_t | NESI] = Σ_k k · P(Y_t = k | NESI),
Var[Y_t | NESI] = E[Y_t² | NESI] − (E[Y_t | NESI])².

Predicted scores are mapped back to the scale's native direction (e.g., pGCS = 15 − E[Y_t | NESI]).

### Gaussian worm approximation

For visualization, we approximate each scale's conditional distribution as a univariate Gaussian in raw-score units,

Y | NESI ~ N(μ_Y(NESI), σ_Y²(NESI)),

with μ_Y(NESI) = E[Y | NESI] and σ_Y²(NESI) = max(Var[Y | NESI], σ_floor²). The SD floor σ_floor = 0.35 prevents the worm from collapsing to zero width at NESI extremes where the ordinal-logistic probabilities concentrate on a single category. Pairs of scales are assumed conditionally independent given NESI, so their joint conditional is the product of the marginals.

### Marginal joint density and centile contours

To sweep out the full "worm," we marginalize the joint conditional over a uniform prior on NESI ∈ [−3, +3]:

p(x, y) = (1 / 6) · ∫_{−3}^{+3} p(x | NESI) · p(y | NESI) dNESI.

This is evaluated on a fine (x, y) grid. Credible regions are highest-density regions (HDRs): for each target mass α ∈ {0.80, 0.95} we find the density threshold c_α such that ∫∫_{p(x,y) ≥ c_α} p(x, y) dx dy = α, and draw the corresponding iso-density contour. The region inside the 95% contour is treated as the plausible support of the model; points outside are rendered white (implausible under the fit).

### Posterior expected NESI

Within the plausible support, each (x, y) location is colored by the posterior expected NESI,

E[NESI | x, y] = ∫ NESI · p(x | NESI) · p(y | NESI) dNESI / ∫ p(x | NESI) · p(y | NESI) dNESI,

under the same uniform prior. Colormap fill alpha is reduced (~0.7) so the red–blue field sits behind the centile contours and the spine, rather than competing with them.

### Spines and NESI markers

The black curves are spines of the worm: the parametric curve (E[X | NESI], E[Y | NESI]) as NESI varies continuously. White-edged dots at NESI ∈ {−2, −1, 0, +1, +2} are placed along each spine so cross-scale correspondences can be read off directly. Numeric NESI annotations are shown only on one exemplar panel (bottom-right, ICANS vs NESI) to avoid label clutter elsewhere.

### Rater-vs-rater reproducibility (supplementary)

For the supplementary figure we compute the same joint density with X and Y set to the *same* scale, using two independent draws from that scale's ordinal-logistic conditional on NESI. Under the assumption that the fitted noise model captures inter-rater variability, this visualizes the expected agreement between two independent raters scoring the same patient. Concentration along the line y = x indicates high reproducibility; dispersion orthogonal to y = x indicates inter-rater noise.

---

## Main figure caption (figure_main.png)

**NESI and the clinical scales.** 4 × 4 matrix of worm-model fits. *Diagonal cells* show each clinical scale as a function of NESI (x-axis = NESI; y-axis = scale). *Off-diagonal cells* show the joint credible region of each pair of clinical scales under the shared-NESI noise model (x-axis = column scale; y-axis = row scale). Nested black contour lines mark the 80 % and 95 % highest-density regions of the marginal joint p(X, Y) obtained by integrating over a uniform NESI prior on [−3, +3]. Fill color inside the 95 % contour is the posterior expected NESI E[NESI | X, Y] (blue → red = better → worse clinical state); outside is white (implausible under the fit). Black curves are the spine (E[X | NESI], E[Y | NESI]); white-edged dots mark NESI = −2, −1, 0, +1, +2 along each spine, with numeric labels shown on the bottom-right panel as an exemplar. Axis labels appear only on the left column and bottom row.

## Supplementary figure caption (figure_supp_rater_noise.png)

**Rater-vs-rater reproducibility implied by the noise model.** Self-vs-self joint distribution on each clinical scale when two independent raters score the same patient, obtained by setting X = Y = scale in the same machinery as the main figure's off-diagonals. 80 % and 95 % credible regions; y = x dashed for reference. Fill color encodes posterior E[NESI | X, Y] (blue → red = better → worse). Tight concentration along y = x indicates small model-implied inter-rater noise; width orthogonal to y = x indicates expected disagreement.
