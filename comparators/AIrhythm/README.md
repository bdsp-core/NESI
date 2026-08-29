# AIrhythm — feature-engineered EEG/ECG coma-prognostication model

**Role in this repository:** external comparator for the NESI (formerly *YAMA*) EEG severity /
mortality models. AIrhythm is a hand-engineered, gradient-boosting baseline that predicts
neurological outcome from multi-hour continuous EEG + ECG. It is the "classical features +
tree ensemble" counterpoint to NESI's foundation-model-plus-contrastive-encoder pipeline.

The code in [`team_code/`](team_code/) is the **unmodified** entry as submitted; nothing in this
directory is wired into the NESI pipeline yet. See
[Using AIrhythm as a NESI comparator](#using-airhythm-as-a-nesi-comparator) for what an
adapter has to supply.

---

## Provenance

| | |
|---|---|
| Origin | [George B. Moody PhysioNet/CinC Challenge 2023](https://moody-challenge.physionet.org/2023/) — *Predicting Neurological Recovery from Coma After Cardiac Arrest* |
| Team | AIrhythm |
| Author | Morteza Zabihi (`morteza.zabihi@gmail.com`) |
| Version string | `v.9.1.1` (printed by `build_models_AIrhythm`) |
| Copyright | © 2023 The General Hospital Corporation |
| License | CC BY-NC 4.0 — [`team_code/LICENSE`](team_code/LICENSE) (same license as this repository) |
| Training data | [I-CARE v2.0](https://physionet.org/content/i-care/2.0/) — 607 patients, 7 hospitals |
| Source archive | `AIrhythm.tar.gz`, extracted verbatim into `team_code/` |

Attribution and the NonCommercial restriction travel with this code. Keep
`team_code/LICENSE`, `team_code/AUTHORS.txt`, and `team_code/README.md` in place, and cite
Zabihi if AIrhythm numbers appear in a manuscript.

---

## What it predicts

Per patient, three outputs (written by `run_model.py` to `<pid>/<pid>.txt`):

1. **`outcome`** — binary. `1` = *poor* neurological outcome (CPC 3–5), `0` = *good* (CPC 1–2),
   assessed at hospital discharge.
2. **`outcome_probability`** — ensemble probability of poor outcome.
3. **`cpc`** — Cerebral Performance Category, continuous, clipped to [1, 5].

> **Label caveat for NESI comparisons.** The positive class is *poor neurological outcome*, not
> death. CPC 5 is death, but CPC 3–4 (severe disability / vegetative state) are also positives.
> The NESI mortality work (`mortality_analysis/`, `NESI/DeathPrediction_NESIvsGCS/`) predicts
> in-hospital death. These are different endpoints and a head-to-head needs one shared label —
> see [below](#label-alignment).

The official challenge metric (`team_code/evaluate_model.py :: compute_challenge_score`) is
**true-positive rate for poor outcome at a false-positive rate ≤ 0.05**, pooled across hospitals
via per-hospital confusion matrices. That operating point — "how many poor-outcome patients can
you flag while almost never mislabelling someone who would have recovered" — is the clinically
meaningful one for prognostication and is worth reusing for any NESI comparison.

---

## Input contract

AIrhythm reads the PhysioNet folder layout directly and is fairly strict about it:

```
data_folder/
  0284/
    0284.txt                 # patient metadata (Age, Sex, ROSC, OHCA, Shockable Rhythm, TTM, Outcome, CPC, Hospital)
    0284_001_004_EEG.hea     # WFDB header: Utility frequency, <start time>, #Length
    0284_001_004_EEG.mat
    0284_001_004_ECG.hea     # optional, must match the EEG in duration
    0284_001_004_ECG.mat
```

Hard requirements, all enforced in `team_code.py :: get_features`:

- **All 17 EEG electrodes present**: `Fp1 F3 C3 P3 Fp2 F4 C4 P4 Fz Cz Pz T3 T5 T4 T6 O1 O2`.
  A recording missing any one of them is silently skipped.
- **≥ 15 minutes** per recording, and start time **< 72 h** after ROSC (parsed from the `.hea`
  start-time field; the loop `break`s at the first recording past 72 h).
- **Not flat/DC** — `detect_dc` rejects a recording if more than a third of its 180-s windows
  have a zero-sum first difference.
- ECG is optional. When it is absent the 16 ECG columns become `NaN`; when it is present but
  unusable (DC, duration mismatch, < 50 detected R-peaks) they become `0`.

Patient metadata splits two ways:

- **Predictors** (appended to every window's feature vector): age, ROSC time, OHCA flag,
  shockable rhythm.
- **Confounders** (never fed to the classifier): sex, hospital. These are used *only* to build
  the cross-validation partitions — see [Robustness](#robustness-three-cv-partitions).
- `TTM` is read and then discarded.

A patient with no usable EEG still gets a prediction: the feature matrix is filled with zeros
and only the four metadata predictors carry signal.

---

## Pipeline

### 1. Preprocessing (`preprocess_data`)

Notch at the utility frequency **and** at half of it → band-pass **0.1–45 Hz** → resample to
**128 Hz** (125 Hz if the original rate is odd) via `resample_poly` → per-channel z-score.

### 2. Montage

The 17 electrodes are recombined into **21 bipolar derivations**. Six of them carry the main
univariate feature bank (`Fz–Cz`, `C3–P3`, `C4–P4`, `Fp1–Fp2`, `T3–T4`, `Cz–Pz`); the rest feed
the connectivity, frontal, and posterior/anterior blocks. `Fp1–Fp2` and `T3–T4` follow
[Resuscitation 2023, 10.1016/j.resuscitation.2023.109817](https://doi.org/10.1016/j.resuscitation.2023.109817).

### 3. Windowing and artifact handling

Non-overlapping **180-second** windows. A window whose amplitude exceeds **5 × IQR** of the
whole channel, or whose variance is ~0, is replaced by a zero feature vector rather than
dropped — so the time axis stays intact. Skipped *recordings* advance a `delay` counter so
that gaps in coverage are preserved in the timestamp vector.

### 4. Per-window features — 382 columns

| Block | Dims | Channels | What it is |
|---|---:|---|---|
| Univariate EEG bank | **258** | 6 bipolar × 43 | see breakdown below |
| Phase connectivity | **24** | pairs of the 6 | PLV + phase-lag index in 16–25 Hz and 0.5–4 Hz |
| Correlation eigenvalues | **21** | all 21 | `eigh(corrcoef)` of the 21-channel window |
| Covariance eigenvalues | **21** | all 21 | `eigh(cov)` of the same |
| Riemannian autocorrelation | **15** | all 21 | top-5 eigenvalues of lagged autocorrelation matrices at 2 s, 8 s, 32 s |
| Path signature | **6** | Fz–Cz | depth-2 `esig` signature of a 3-D delay embedding (lag 0.3 s), summarized by 6 quantiles of its increments |
| Compression distance | **1** | Fz–Cz | normalized compression distance (SAX symbolization → gzip) between *consecutive* 60-s windows — a drift/stationarity measure |
| Frontal depth-of-anesthesia | **12** | Fp1–T3, Fp2–T4, Fp1–Cz, Fp2–Cz | ×3 each: BIS-style `log10(P₃₀₋₄₇/P₁₁₋₂₀)`, `log(P₃₀₋₄₂.₅/P₆₋₁₂)`, log–log PSD slope |
| Posterior/anterior | **4** | (Fp1–Fp2 ÷ O1–O2), (Fp1–T3 ÷ Fp2–T4) | ×2 each: alpha-power ratio, spectral-slope ratio |
| ECG | **16** | first ECG lead | HRV (10) + AFEv atrial-fibrillation evidence (3) + shockable-vs-nonshockable morphology (3) |
| *EEG/ECG subtotal* | *378* | | |
| Patient metadata | **4** | — | age, ROSC, OHCA, shockable rhythm |
| **Total** | **382** | | |

The 43-feature univariate bank (`featurize_dynamic_winodows_eeg`), per channel per window:

- Hjorth activity, mobility, complexity — **3**
- Power ≤ 1 Hz; (4–12)/(12–30) ratio; (4–12)/(8–35) ratio; alpha/delta ratio — **4**
- IQR of slow-wave (0.5–4 Hz) zero-crossing intervals — **1**
- db4 wavelet, 5 levels: mean and variance of each detail coefficient — **10**
- IQR of 16–25 Hz zero-crossing intervals — **1**
- STFT summary of the 0–40 Hz log-spectrogram: peak frequency, variance, and the mean and std of
  the instantaneous frequency of its band-summed analytic signal — **4**
- Spindle count: 13 Hz Morlet wavelet (12 cycles), events above 0.25 normalized power — **1**
- Burg AR(10): residual variance **1** + coefficients **10**
- Seasonal autocorrelation at 0.3 s, 0.8 s, 1.5 s — **3**
- Spectral edge frequencies at 50/70/80/90/95 % — **5**

The design leans hard on **burst-suppression, background continuity, reactivity-adjacent, and
depth-of-anesthesia** proxies — exactly the physiology clinicians read in post-arrest EEG — plus
an ECG channel that NESI does not use at all.

### 5. Temporal pooling (`class_model.create_temporal_datasets`)

Window indices are converted to hours (`index × 3 min`), zeroed at the first valid window, then
chopped into consecutive **`h`-hour blocks** across the first 72 h. Each block is collapsed to a
single row by one of:

| `prepop` | Pooling |
|---|---|
| `q88` / `q89` | 88th / 89th percentile per column |
| `combine`, `combine1` | skew **‖** 89th percentile (doubles the dimension) |
| `mean`, `diff_mean` | mean, or mean ÷ std |

A high percentile rather than a mean is the deliberate choice: it tracks the *worst* few minutes
in the block, which is where prognostic signal lives. A sinusoidal **position encoding** is then
added to each block row so the classifier knows how far post-arrest the block sits.

`create_temporal_datasets_daily` (used by the `stacking2` config) builds three parallel views of
each block — the `h`-hour window, a 12-hour window, and an 18-hour window, all sharing a start —
so one model sees three time-scales at once.

### 6. Classifier zoo

Thirteen configurations (`build_models_AIrhythm`), each crossed with a CV partition:

| Base learner | `h` (hours) | Pooling | Partitions |
|---|---|---|---|
| CatBoost (1500 iters, lr 0.003, depth 6) | 6 | `q89` | normal / cluster / PS |
| CatBoost | 5 | `q89` | normal / cluster / PS |
| `stacking1` — MLP + SVM + CatBoost + ExtraTrees + LDA → MLP meta | 6 | `q88` | normal / cluster / PS |
| CatBoost | 6.9 | `combine` | normal / cluster / PS |
| `stacking2` — three CatBoosts (one per time-scale view) → logistic-regression meta, 5 inner folds | 6.9 | `combine1` | cluster |

Each config keeps **every** fold's model rather than refitting on all data: 5 (normal) + 3
(cluster) + 4 (PS) = 12 models for each of the first four rows, plus 3 for `stacking2` →
**51 fitted classifiers**, each paired with an `XGBRegressor` CPC head. Training classes are
balanced by undersampling the majority class before each fit.

### 7. Robustness: three CV partitions

This is the part worth stealing conceptually. Instead of one random split, folds are drawn three
different ways, all from the **confounders** (sex, hospital):

- **`cv` (normal)** — plain 5-fold `KFold` over patients.
- **`cv_cluster`** — k-means (k = 3) on normalized confounders; each cluster is held out in turn.
  Folds therefore differ systematically by site/demographics.
- **`cv_ps`** — a logistic propensity model predicts outcome *from confounders alone*; patients
  are split into quartiles of that propensity and each quartile is held out in turn.

Ensembling across all three forces the ensemble to be less dependent on any single site or
demographic stratum — a direct answer to the challenge's hidden-hospital test set, and a
sensible guard for our own multi-site EEG data.

### 8. Inference (`helper_run_model_AIrhythm`)

For one patient:

1. Each of the 51 models scores every `h`-hour block.
2. Within a model, per-block probabilities are combined by a **quantile-weighted average** with
   weights `0.2 / 0.5 / 1 / 1.5 / 1.8 / 2` rising across the 10/30/50/70/90th percentiles — the
   worst blocks dominate. (Note: this is *not* a plain mean; a patient who looks bad for a few
   hours is scored badly overall.)
3. Probabilities are averaged across all 51 models → `outcome_probability`, thresholded at 0.5.
4. CPC = median of the CPC-head predictions from the models whose binary vote matched the
   ensemble label.

---

## Running it

Python **3.8** (the Dockerfile pins `python:3.8.8`); the pinned deps in
[`team_code/requirements.txt`](team_code/requirements.txt) include `catboost==1.2`,
`esig==0.9.8.3`, `mne==1.4.2`, `pyts==0.13.0`, `xgboost==1.7.6`. `esig` is the fragile one — it
needs a working C toolchain and does not build cleanly on recent Python. Use the container.

```bash
cd comparators/AIrhythm/team_code

# native
pip install -r requirements.txt
python train_model.py  <data_folder> <model_folder>            # writes <model_folder>/models.sav
python run_model.py    <model_folder> <data_folder> <outputs>
python evaluate_model.py <labels_folder> <outputs>

# or containerized (recommended — matches the challenge environment)
docker build -t airhythm .
docker run -it -v ~/data:/challenge/data -v ~/model:/challenge/model \
                -v ~/outputs:/challenge/outputs airhythm bash
```

Helper scripts for building held-out sets: `truncate_data.py` (cut recordings at an hour limit),
`remove_labels.py`, `remove_data.py`.

**Cost.** Training touches ~25,800 EEG files for the 72-hour configuration; feature extraction is
the bottleneck (single-threaded Python per window, with `esig` and gzip in the inner loop), and
51 model fits follow. Budget many hours on a large box, and extract features once to disk before
sweeping configurations.

---

## Using AIrhythm as a NESI comparator

Three gaps have to be closed before the numbers mean anything.

### Data format

AIrhythm wants continuous multi-hour recordings in PhysioNet WFDB layout with an hour offset in
the header. Our HEEDB data are 10-minute segments (`yama/segment_index.csv` maps each segment to
its continuous source EEG). An adapter has to either:

- **(a)** reconstruct continuous stretches from the source EEGs and synthesize `.hea` files with
  correct start times — closest to how AIrhythm was designed, and the only way its temporal
  pooling and 72-hour structure do anything; or
- **(b)** run it per 10-minute segment, which reduces it to a single 3-minute-window feature
  bank plus one pooling step. Cheaper, but it discards the temporal machinery and is a weaker
  comparator. If we go this way, say so explicitly in the write-up.

Either way, the 17-electrode requirement and the `< 72 h` gate need relaxing or re-anchoring —
our cohorts have no ROSC reference time.

### Label alignment

Retrain, don't reuse. The shipped design targets CPC 3–5 at discharge in post-arrest coma.
For a NESI comparison, refit the same feature bank and classifier zoo against **in-hospital
death**, the label already assembled in
[`mortality_analysis/Cohort/YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv`](../../mortality_analysis/Cohort/YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv).
The CPC regression head has no analogue in our data and should be dropped.

Also drop or replace the four metadata predictors (age, ROSC, OHCA, shockable rhythm): three of
them are cardiac-arrest-specific. Keeping age alone is the honest minimum, and the EEG-only
variant is the cleaner comparison against NESI.

### Comparison design

Match the existing mortality analysis so the curves are plottable together:

- Train / evaluate: [`NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Train_Death_Prediction_with_NESI_vsGCS.py`](../../NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Train_Death_Prediction_with_NESI_vsGCS.py)
- Plot: [`NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Test_results_plot_GCSvsNESI.py`](../../NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Test_results_plot_GCSvsNESI.py)
- Correlation-style analysis: [`mortality_analysis/NESI_Corelation_with_Death.py`](../../mortality_analysis/NESI_Corelation_with_Death.py)

That figure currently reports longitudinal AUROC for NESI vs GCS (≈0.82 vs 0.76 at 20 h);
AIrhythm slots in as a third curve. Report AUROC *and* the challenge-style TPR at FPR ≤ 0.05 —
the second is where a feature-engineered ensemble and a foundation-model embedding tend to
diverge most.

Use the same patient-level splits as the NESI models. AIrhythm's own three-partition CV is a
property of *its* training procedure, not a shared evaluation protocol — don't let it pick its
own folds and then compare test numbers.

---

## Files

| File | Role |
|---|---|
| `team_code/team_code.py` | Entry points, `get_features`, preprocessing, montage, train/run orchestration |
| `team_code/utilities_AIrhythm.py` | Univariate EEG bank, connectivity, eigen/Riemannian features, ECG feature assembly |
| `team_code/class_model.py` | Temporal pooling, position encoding, NaN handling, class balancing, learners, inference |
| `team_code/class_robust.py` | The three CV partitions (`cv`, `cv_cluster`, `cv_ps`) and metrics |
| `team_code/utility_class_ECG_v1.py` | ECG/HRV feature library (AFEv, dRR metrics, P-wave checks, morphology) |
| `team_code/utility_class_qrs_detection.py` | QRS detector |
| `team_code/Compressors_features.py` | SAX + gzip normalized compression distance |
| `team_code/tangent_signiture.py` | Delay embedding + `esig` path signature |
| `team_code/frontal_features.py` | BIS-style frontal / depth-of-anesthesia features |
| `team_code/post_ant.py` | Posterior-vs-anterior alpha and slope ratios |
| `team_code/helper_code.py`, `train_model.py`, `run_model.py`, `evaluate_model.py` | Challenge-provided harness (unmodified) |
| `team_code/truncate_data.py`, `remove_data.py`, `remove_labels.py` | Challenge dataset utilities |
| `team_code/LICENSE`, `AUTHORS.txt`, `README.md` | Original license and author statement — keep with the code |
