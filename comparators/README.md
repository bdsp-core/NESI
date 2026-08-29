# Comparators

External / baseline models kept alongside NESI so that head-to-head comparisons are reproducible
from one place. Code here is **not** part of the NESI pipeline: each subdirectory holds a
third-party model as received, plus documentation of what it does and what an adapter to our data
has to supply.

| Model | Task it was built for | Approach | Docs |
|---|---|---|---|
| [`AIrhythm/`](AIrhythm/) | Neurological outcome (CPC) after cardiac-arrest coma — PhysioNet/CinC Challenge 2023 | Hand-engineered EEG + ECG features → CatBoost / stacked-ensemble zoo over three CV partitions | [`AIrhythm/README.md`](AIrhythm/README.md) |

Each comparator keeps its own upstream `LICENSE` and author attribution. AIrhythm is
CC BY-NC 4.0, © 2023 The General Hospital Corporation — the same license as this repository.
