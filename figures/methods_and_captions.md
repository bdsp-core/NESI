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

This is evaluated on a fine (x, y) grid. Credible regions are highest-density regions (HDRs): for each target mass α ∈ {0.5, 0.8, 0.95} we find the density threshold c_α such that ∫∫_{p(x,y) ≥ c_α} p(x, y) dx dy = α, and draw the corresponding iso-density contour. The region inside the 95% contour is treated as the plausible support of the model; points outside are rendered white (implausible under the fit).

### Posterior expected NESI

Within the plausible support, each (x, y) location is colored by the posterior expected NESI,

E[NESI | x, y] = ∫ NESI · p(x | NESI) · p(y | NESI) dNESI / ∫ p(x | NESI) · p(y | NESI) dNESI,

under the same uniform prior. This provides a continuous readout of which NESI values are consistent with each score combination.

### Rater-vs-rater diagonals (Figure 2)

For the diagonal cells of Figure 2 we compute the same joint density with X and Y set to the *same* scale using two independent draws from that scale's ordinal logistic conditional on NESI. Under the assumption that the fitted noise model captures inter-rater variability, this visualizes the expected agreement between two independent raters scoring the same patient. Concentration along the line y = x indicates high reproducibility; dispersion orthogonal to y = x indicates inter-rater noise.

### Spines and NESI markers

The black curves in both figures are spines of the worm: the parametric curve (E[X | NESI], E[Y | NESI]) as NESI varies continuously. White-edged dots at NESI ∈ {−2, −1, 0, +1, +2} are placed along each spine so cross-scale correspondences can be read off directly (e.g., "NESI = +1 corresponds to pGCS ≈ 7, pRASS ≈ −3.7, pCAMS ≈ 6.8, pICANS ≈ 2.9").

---

## Figure 1 caption

**Predicted clinical scale score as a function of NESI, by scale.** Each panel shows the fitted relationship between NESI (x-axis) and one clinical scale (y-axis). The colored region is the 95% credible region of Y | NESI under the proportional-odds ordinal-logistic fit (Gaussian-approximated for display), filled by NESI value (blue → red = low → high NESI). Nested black contours mark the 50%, 80%, and 95% centiles of p(Y, NESI) under a uniform NESI prior on [−3, +3]. The black curve is the posterior mean E[Y | NESI]; white-edged dots along it mark NESI = −2, −1, 0, +1, +2. Panel titles give the Train-split sample size, the fitted slope β, and the Spearman correlation between observed score and NESI.

## Figure 2 caption

**Pairwise structure of the clinical scales under the shared-NESI noise model.** Each cell shows the joint 50/80/95% credible region of a pair of scale scores under the model Y = f(NESI) + Gaussian noise, with conditional independence between the two scales given NESI and a uniform NESI prior on [−3, +3]. *Off-diagonal cells* plot two different scales against each other; the worm shape traces the set of score combinations that are jointly plausible under the fit. *Diagonal cells* plot a single scale against itself, treated as two independent raters of the same patient; their tightness around y = x reflects the model-implied rater-to-rater reproducibility. Fill color inside the 95% contour is the posterior expected NESI E[NESI | X, Y] (blue → red = low → high NESI). Outside the 95% contour is white (implausible under the model). Black curves are worm spines (E[X | NESI], E[Y | NESI]); white-edged dots mark NESI = −2, −1, 0, +1, +2. Axis labels appear only on the left column and bottom row.
