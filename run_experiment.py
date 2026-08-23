"""Run the minimal structural-prior mismatch study in keypoint space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class Sample:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray


@dataclass
class Result:
    object_type: str
    prior: str
    train_size: int
    seed: int
    prediction_error: float
    structural_violation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("structural_prior_experiment/runs/main"))
    parser.add_argument("--train_sizes", nargs="+", type=int, default=[100, 500, 1000, 5000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--prior_weight", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)


def make_transition(object_type: str, rng: np.random.Generator) -> Sample:
    action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    translation = np.array([0.16 * action[0], 0.16 * action[1]], dtype=np.float32)
    angle = float(0.35 * action[2])

    if object_type == "rigid":
        local = np.array([[-0.25, -0.12], [-0.25, 0.12], [0.25, -0.12], [0.25, 0.12]], dtype=np.float32)
        center = rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
        state = local @ rotation(float(rng.uniform(-math.pi, math.pi))).T + center
        next_state = state @ rotation(angle).T + translation
        return Sample(state, action, next_state.astype(np.float32))

    if object_type != "articulated":
        raise ValueError(f"Unknown object type: {object_type}")

    base = np.array([[-0.35, -0.10], [-0.35, 0.10], [0.0, -0.10], [0.0, 0.10]], dtype=np.float32)
    child = np.array([[0.0, -0.10], [0.0, 0.10], [0.35, -0.10], [0.35, 0.10]], dtype=np.float32)
    center = rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
    base_angle = float(rng.uniform(-math.pi, math.pi))
    base_rotation = rotation(base_angle)
    state = np.concatenate([base @ base_rotation.T, child @ base_rotation.T], axis=0) + center
    joint_angle = 0.75 * action[2] + 0.10 * action[0]
    next_base = base @ rotation(base_angle + angle).T + center + translation
    next_child_local = child @ rotation(base_angle + angle + joint_angle).T
    next_state = np.concatenate([next_base, next_child_local + center + translation], axis=0)
    return Sample(state.astype(np.float32), action, next_state.astype(np.float32))


def make_dataset(object_type: str, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    samples = [make_transition(object_type, rng) for _ in range(size)]
    states = torch.from_numpy(np.stack([sample.state for sample in samples]))
    actions = torch.from_numpy(np.stack([sample.action for sample in samples]))
    next_states = torch.from_numpy(np.stack([sample.next_state for sample in samples]))
    return states, actions, next_states


class DynamicsNet(nn.Module):
    def __init__(self, keypoints: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = keypoints * 2 + 3
        output_dim = keypoints * 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([states.flatten(1), actions], dim=1)).reshape_as(states)


def pair_distances(states: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.stack([torch.linalg.vector_norm(states[:, left] - states[:, right], dim=1) for left, right in pairs], dim=1)


def prior_pairs(prior: str, object_type: str) -> list[tuple[int, int]]:
    keypoint_count = 4 if object_type == "rigid" else 8
    all_pairs = [(left, right) for left in range(keypoint_count) for right in range(left + 1, keypoint_count)]
    if prior == "none":
        return []
    if prior == "global":
        return all_pairs
    if prior == "part":
        if object_type == "rigid":
            return all_pairs
        return [
            (left, right)
            for offset in (0, 4)
            for left in range(offset, offset + 4)
            for right in range(left + 1, offset + 4)
        ]
    raise ValueError(f"Unknown prior: {prior}")


def train_model(
    object_type: str,
    prior: str,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    prior_weight: float,
    hidden_dim: int,
) -> DynamicsNet:
    states, actions, targets = train_data
    model = DynamicsNet(keypoints=train_data[0].shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pairs = prior_pairs(prior, object_type)
    for _ in range(epochs):
        permutation = torch.randperm(len(states))
        for indices in permutation.split(batch_size):
            prediction = model(states[indices], actions[indices])
            prediction_loss = torch.mean((prediction - targets[indices]) ** 2)
            if pairs:
                predicted_distances = pair_distances(prediction, pairs)
                current_distances = pair_distances(states[indices], pairs)
                prior_loss = torch.mean((predicted_distances - current_distances) ** 2)
            else:
                prior_loss = torch.zeros(())
            loss = prediction_loss + prior_weight * prior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: DynamicsNet, object_type: str, prior: str, test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float]:
    states, actions, targets = test_data
    with torch.no_grad():
        predictions = model(states, actions)
        prediction_error = torch.mean(torch.linalg.vector_norm(predictions - targets, dim=2)).item()
        # Keep this metric fixed across methods: it measures physically valid
        # intra-part rigidity, including for models trained with no prior.
        consistency_pairs = prior_pairs("part", object_type)
        violation = torch.mean(
            torch.abs(pair_distances(predictions, consistency_pairs) - pair_distances(states, consistency_pairs))
        ).item()
    return prediction_error, violation


def save_results(results: list[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    (output_dir / "metrics.json").write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def plot_results(results: list[Result], output_dir: Path) -> None:
    grouped: dict[tuple[str, str, int], list[Result]] = {}
    for result in results:
        grouped.setdefault((result.object_type, result.prior, result.train_size), []).append(result)

    def curve(object_type: str, prior: str, metric: str) -> tuple[list[int], list[float]]:
        values = []
        for (current_type, current_prior, train_size), entries in grouped.items():
            if current_type == object_type and current_prior == prior:
                values.append((train_size, float(np.mean([getattr(entry, metric) for entry in entries]))))
        values.sort()
        return [value[0] for value in values], [value[1] for value in values]

    for metric, ylabel, filename in [
        ("prediction_error", "Mean keypoint error", "prediction_error.png"),
        ("structural_violation", "Mean distance violation", "structural_violation.png"),
    ]:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for axis, object_type in zip(axes, ["rigid", "articulated"]):
            for prior in ["none", "global", "part"]:
                train_sizes, values = curve(object_type, prior, metric)
                axis.plot(train_sizes, values, marker="o", label=prior)
            axis.set_title(object_type.capitalize())
            axis.set_xscale("log")
            axis.set_xlabel("Training transitions")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if min(args.train_sizes) <= 0 or min(args.seeds) < 0:
        raise ValueError("train sizes must be positive and seeds must be non-negative")
    results: list[Result] = []
    for object_type in ["rigid", "articulated"]:
        test_data = make_dataset(object_type, args.test_size, seed=100000 + (0 if object_type == "rigid" else 1))
        for train_size in args.train_sizes:
            for seed in args.seeds:
                train_data = make_dataset(object_type, train_size, seed=seed + (1000 if object_type == "articulated" else 0))
                for prior in ["none", "global", "part"]:
                    set_seed(seed)
                    model = train_model(
                        object_type, prior, train_data, args.epochs, args.batch_size,
                        args.learning_rate, args.prior_weight, args.hidden_dim,
                    )
                    error, violation = evaluate(model, object_type, prior, test_data)
                    result = Result(object_type, prior, train_size, seed, error, violation)
                    results.append(result)
                    print(asdict(result))
    save_results(results, args.output_dir)
    plot_results(results, args.output_dir)
    print(f"Saved {len(results)} runs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()