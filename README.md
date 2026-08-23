# Structural Prior Mismatch Experiment

This directory contains a self-contained keypoint-space experiment for the
structural-prior study. It deliberately avoids RGB perception and the old
Prism cube-pushing pipeline.

The experiment compares three models on rigid and articulated planar objects:

- `none`: prediction loss only;
- `global`: rigidity loss on every keypoint pair;
- `part`: rigidity loss only within the known rigid parts.

Run a quick smoke test from any directory:

```bash
/home/yjxie/miniconda3/envs/vla/bin/python /home/yjxie/structural_prior_experiment/run_experiment.py \
  --output_dir /home/yjxie/structural_prior_experiment/runs/smoke \
  --train_sizes 32 64 --seeds 0 --epochs 10
```

Run the initial study:

```bash
/home/yjxie/miniconda3/envs/vla/bin/python /home/yjxie/structural_prior_experiment/run_experiment.py \
  --output_dir /home/yjxie/structural_prior_experiment/runs/main \
  --train_sizes 100 500 1000 5000 --seeds 0 1 2 3 4 --epochs 150
```

The script writes `metrics.csv`, `metrics.json`, and plots for prediction
error, rigidity violation, and data scaling. The simulator uses analytic
planar motion: a rigid object receives a shared SE(2)-like transform, while an
articulated object has a fixed base part and a revolute joint whose angle is
action-dependent.