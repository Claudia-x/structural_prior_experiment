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
    mass: float
    friction: float


@dataclass
class Result:
    object_type: str
    prior: str
    train_size: int
    seed: int
    prior_weight: float
    prediction_error: float
    structural_violation: float
    planning_success: float


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
    parser.add_argument(
        "--prior_weights",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 0.3, 1.0, 2.0],
        help="Weights scanned for global and part priors; none always uses weight 0.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--action_noise", type=float, default=0.08)
    parser.add_argument("--observation_noise", type=float, default=0.01)
    parser.add_argument("--planning_episodes", type=int, default=100)
    parser.add_argument("--planning_horizon", type=int, default=3)
    parser.add_argument("--planning_candidates", type=int, default=27)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)


def physical_step(
    object_type: str,
    state: np.ndarray,
    action: np.ndarray,
    mass: float,
    friction: float,
) -> np.ndarray:
    dt = 0.12
    force = action[:2] * 1.5
    displacement = dt * dt * force / mass
    displacement *= max(0.0, 1.0 - friction * 0.35)
    angle = float(dt * dt * action[2] / (mass * 0.18))
    angle *= max(0.0, 1.0 - friction * 0.25)
    center = state.mean(axis=0)
    translated = state + displacement

    if object_type == "rigid":
        return ((translated - center) @ rotation(angle).T + center).astype(np.float32)

    if object_type != "articulated":
        raise ValueError(f"Unknown object type: {object_type}")

    base = translated[:4]
    child = translated[4:]
    base_center = base.mean(axis=0)
    base = (base - base_center) @ rotation(angle).T + base_center
    child_center = child.mean(axis=0)
    joint_angle = 0.65 * angle + 0.04 * action[2]
    child = (child - child_center) @ rotation(joint_angle).T + child_center
    return np.concatenate([base, child], axis=0).astype(np.float32)


def make_transition(
    object_type: str,
    rng: np.random.Generator,
    action_noise: float = 0.0,
    observation_noise: float = 0.0,
) -> Sample:
    clean_action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    action = clean_action + rng.normal(0.0, action_noise, size=3).astype(np.float32)
    mass = float(rng.uniform(0.5, 2.0))
    friction = float(rng.uniform(0.15, 0.65))

    if object_type == "rigid":
        local = np.array([[-0.25, -0.12], [-0.25, 0.12], [0.25, -0.12], [0.25, 0.12]], dtype=np.float32)
        center = rng.uniform(-0.15, 0.15, size=2).astype(np.float32)
        state = local @ rotation(float(rng.uniform(-math.pi, math.pi))).T + center
    else:
        base = np.array([[-0.35, -0.10], [-0.35, 0.10], [0.0, -0.10], [0.0, 0.10]], dtype=np.float32)
        child = np.array([[0.0, -0.10], [0.0, 0.10], [0.35, -0.10], [0.35, 0.10]], dtype=np.float32)
        center = rng.uniform(-0.15, 0.15, size=2).astype(np.float32)
        base_angle = float(rng.uniform(-math.pi, math.pi))
        state = np.concatenate([base @ rotation(base_angle).T, child @ rotation(base_angle).T], axis=0) + center
    next_state = physical_step(object_type, state, clean_action, mass, friction)
    observed_state = state + rng.normal(0.0, observation_noise, state.shape).astype(np.float32)
    return Sample(observed_state.astype(np.float32), action, next_state, mass, friction)


def make_dataset(
    object_type: str,
    size: int,
    seed: int,
    action_noise: float = 0.0,
    observation_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = [make_transition(object_type, rng, action_noise, observation_noise) for _ in range(size)]
    states = torch.from_numpy(np.stack([sample.state for sample in samples]))
    actions = torch.from_numpy(np.stack([sample.action for sample in samples]))
    next_states = torch.from_numpy(np.stack([sample.next_state for sample in samples]))
    masses = np.asarray([sample.mass for sample in samples], dtype=np.float32)
    frictions = np.asarray([sample.friction for sample in samples], dtype=np.float32)
    return states, actions, next_states, masses, frictions


def save_dataset(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray], path: Path) -> None:
    states, actions, next_states, masses, frictions = data
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states.numpy(),
        actions=actions.numpy(),
        next_states=next_states.numpy(),
        masses=masses,
        frictions=frictions,
    )


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
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    prior_weight: float,
    hidden_dim: int,
) -> DynamicsNet:
    states, actions, targets, _, _ = train_data
    model = DynamicsNet(keypoints=train_data[0].shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pairs = prior_pairs(prior, object_type)
    for _ in range(epochs):
        permutation = torch.randperm(len(states))
        for indices in permutation.split(batch_size):
            prediction = model(states[indices], actions[indices])
            prediction_loss = torch.sum((prediction - targets[indices]) ** 2) / (
                prediction.shape[0] * prediction.shape[1]
            )
            if pairs:
                predicted_distances = pair_distances(prediction, pairs)
                current_distances = pair_distances(states[indices], pairs)
                prior_loss = torch.sum((predicted_distances - current_distances) ** 2) / (
                    predicted_distances.shape[0] * predicted_distances.shape[1]
                )
            else:
                prior_loss = torch.zeros(())
            loss = prediction_loss + prior_weight * prior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: DynamicsNet, object_type: str, prior: str, test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]) -> tuple[float, float]:
    states, actions, targets, _, _ = test_data
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


def planning_success_rate(
    model: DynamicsNet,
    object_type: str,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    episodes: int,
    horizon: int,
    candidate_count: int,
    action_noise: float,
    seed: int,
) -> float:
    states, _, _, masses, frictions = test_data
    rng = np.random.default_rng(seed)
    action_values = np.linspace(-1.0, 1.0, 3)
    candidates = np.asarray(
        [[force_x, force_y, torque] for force_x in action_values for force_y in action_values for torque in action_values],
        dtype=np.float32,
    )[:candidate_count]
    successes = 0
    for index in range(min(episodes, len(states))):
        current = states[index : index + 1]
        for _ in range(horizon):
            rollout_states = current.repeat(len(candidates), 1, 1)
            rollout_actions = torch.from_numpy(candidates)
            predicted = model(rollout_states, rollout_actions)
            scores = torch.linalg.vector_norm(predicted.mean(dim=1), dim=1)
            action = candidates[int(torch.argmin(scores))]
            executed_action = action + rng.normal(0.0, action_noise, size=3).astype(np.float32)
            current = torch.from_numpy(
                physical_step(object_type, current[0].numpy(), executed_action, float(masses[index]), float(frictions[index]))
            ).unsqueeze(0)
        if torch.linalg.vector_norm(current.mean(dim=1)).item() < 0.08:
            successes += 1
    return successes / max(1, min(episodes, len(states)))


def save_results(results: list[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    (output_dir / "metrics.json").write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def plot_results(results: list[Result], output_dir: Path) -> None:
    for metric, ylabel, filename in [
        ("prediction_error", "Mean keypoint error", "prediction_error.png"),
        ("structural_violation", "Mean distance violation", "structural_violation.png"),
    ]:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for axis, object_type in zip(axes, ["rigid", "articulated"]):
            for prior in ["none", "global", "part"]:
                weights = sorted({result.prior_weight for result in results if result.prior == prior})
                for weight in weights:
                    filtered = [
                        result for result in results
                        if result.object_type == object_type and result.prior == prior and result.prior_weight == weight
                    ]
                    values = []
                    for train_size in sorted({result.train_size for result in filtered}):
                        entries = [result for result in filtered if result.train_size == train_size]
                        values.append((train_size, float(np.mean([getattr(entry, metric) for entry in entries]))))
                    label = prior if prior == "none" else f"{prior}, lambda={weight:g}"
                    axis.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=label)
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
    if any(weight < 0.0 for weight in args.prior_weights):
        raise ValueError("prior weights must be non-negative")
    results: list[Result] = []
    data_dir = args.output_dir / "data"
    for object_type in ["rigid", "articulated"]:
        test_data = make_dataset(
            object_type,
            args.test_size,
            seed=100000 + (0 if object_type == "rigid" else 1),
            action_noise=args.action_noise,
            observation_noise=args.observation_noise,
        )
        save_dataset(test_data, data_dir / f"test_{object_type}.npz")
        for train_size in args.train_sizes:
            for seed in args.seeds:
                train_data = make_dataset(
                    object_type,
                    train_size,
                    seed=seed + (1000 if object_type == "articulated" else 0),
                    action_noise=args.action_noise,
                    observation_noise=args.observation_noise,
                )
                save_dataset(train_data, data_dir / f"train_{object_type}_n{train_size}_seed{seed}.npz")
                for prior in ["none", "global", "part"]:
                    weights = [0.0] if prior == "none" else args.prior_weights
                    for prior_weight in weights:
                        set_seed(seed)
                        model = train_model(
                            object_type, prior, train_data, args.epochs, args.batch_size,
                            args.learning_rate, prior_weight, args.hidden_dim,
                        )
                        error, violation = evaluate(model, object_type, prior, test_data)
                        planning = planning_success_rate(
                            model,
                            object_type,
                            test_data,
                            args.planning_episodes,
                            args.planning_horizon,
                            args.planning_candidates,
                            args.action_noise,
                            seed + 500000,
                        )
                        result = Result(object_type, prior, train_size, seed, prior_weight, error, violation, planning)
                        results.append(result)
                        print(asdict(result))
    save_results(results, args.output_dir)
    plot_results(results, args.output_dir)
    print(f"Saved {len(results)} runs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()