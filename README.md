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

Run the normalized baseline study with the default lambda scan:

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

The completed sensitivity run in `runs/lambda_sensitivity/` contains 312
results: 4 training sizes, 3 seeds, 6 nonzero lambda values for each prior,
and the no-prior baseline.

The training objective is

```text
L = L_pred + lambda * L_prior
```

Both terms are normalized: `L_pred` is averaged over keypoints and `L_prior`
is averaged over constrained keypoint pairs. `none` always uses `lambda=0`.
The `global` and `part` methods scan the values passed through
`--prior_weights` so the effect of the prior strength can be separated from
the effect of the prior structure.

Each run also saves reproducible data under `<output_dir>/data/`:

- `test_rigid.npz` and `test_articulated.npz` are fixed test sets;
- `train_<object>_n<size>_seed<seed>.npz` contains each training set;
- every archive has `states`, `actions`, and `next_states` arrays.

Run the lambda sensitivity study:

```bash
/home/yjxie/miniconda3/envs/vla/bin/python /home/yjxie/structural_prior_experiment/run_experiment.py \
  --output_dir /home/yjxie/structural_prior_experiment/runs/lambda_sensitivity \
  --train_sizes 100 500 1000 5000 --seeds 0 1 2 \
  --prior_weights 0.001 0.01 0.1 0.3 1.0 2.0 --epochs 150
```