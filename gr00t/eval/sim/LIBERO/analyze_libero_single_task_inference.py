import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, VLAStepData
from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs, create_eval_env
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one LIBERO task locally with a GR00T checkpoint, record VLM/DiT/action "
            "similarity statistics during inference, and save plots."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_policy_calls", type=int, default=0)
    parser.add_argument("--max_activation_samples", type=int, default=40000)
    parser.add_argument("--sample_values_per_tensor", type=int, default=1024)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def choose_representative_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted({0, count // 2, count - 1})


def parse_layer_index(name: str, pattern: str) -> int | None:
    match = re.search(pattern, name)
    if match:
        return int(match.group(1))
    return None


def cosine_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.float32)
    a = a[:n].float()
    b = b[:n].float()
    return F.cosine_similarity(a, b, dim=-1, eps=1e-8)


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan")
    return float(F.cosine_similarity(a[:n][None], b[:n][None], dim=-1, eps=1e-8).item())


def evenly_sample_tensor(tensor: torch.Tensor, max_values: int) -> np.ndarray:
    flat = tensor.detach().float().cpu().reshape(-1)
    if flat.numel() == 0:
        return np.empty((0,), dtype=np.float32)
    if flat.numel() <= max_values:
        return flat.numpy()
    idx = torch.linspace(0, flat.numel() - 1, steps=max_values).long()
    return flat[idx].numpy()


def build_image_slot_labels(video_obs: dict[str, np.ndarray]) -> list[str]:
    view_names = list(video_obs.keys())
    if not view_names:
        return []
    horizon = int(video_obs[view_names[0]].shape[1])
    return [f"t{t}:{view}" for t in range(horizon) for view in view_names]


def split_chunk_means(values: torch.Tensor, labels: list[str]) -> np.ndarray:
    if not labels:
        return np.empty((0,), dtype=np.float32)
    n = values.shape[0]
    if n == 0:
        return np.zeros((len(labels),), dtype=np.float32)
    if n % len(labels) != 0:
        return np.full((len(labels),), float(values.mean().item()), dtype=np.float32)
    chunk = n // len(labels)
    return values.reshape(len(labels), chunk).mean(dim=1).cpu().numpy().astype(np.float32)


def flatten_action_dict(action: dict[str, np.ndarray], action_keys: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = []
    parts = []
    for key in action_keys:
        arr = action[key][0]
        parts.append(arr)
        labels.extend([f"{key}[{i}]" for i in range(arr.shape[-1])])
    return np.concatenate(parts, axis=-1), labels


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    path: Path,
    matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.4), max(4, matrix.shape[0] * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_line_plot(
    path: Path,
    x: np.ndarray,
    ys: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, values in ys.items():
        ax.plot(x, values, label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if ys:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_histograms(
    path: Path,
    samples: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    bins: int = 80,
) -> None:
    import matplotlib.pyplot as plt

    non_empty = {k: v for k, v in samples.items() if v.size > 0}
    if not non_empty:
        return
    fig, axes = plt.subplots(len(non_empty), 1, figsize=(10, 3 * len(non_empty)), squeeze=False)
    for ax, (name, values) in zip(axes[:, 0], non_empty.items()):
        ax.hist(values, bins=bins, color="#2a6f97", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_boxplot(path: Path, grouped_samples: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    names = [name for name, values in grouped_samples.items() if values.size > 0]
    if not names:
        return
    data = [grouped_samples[name] for name in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 5))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


class SimilarityAnalyzer:
    def __init__(self, policy: Gr00tPolicy, output_dir: Path, args: argparse.Namespace):
        self.policy = policy
        self.output_dir = output_dir
        self.args = args
        self.model = policy.model
        self.image_token_index = self.model.backbone.model.image_token_index
        self.action_keys = list(self.policy.modality_configs["action"].modality_keys)
        self.video_keys = list(self.policy.modality_configs["video"].modality_keys)
        self.device = self.model.device
        self.num_dit_iterations = int(self.model.action_head.num_inference_timesteps)
        self.action_horizon = int(self.model.action_head.action_horizon)

        self.call_steps: list[int] = []
        self.call_rewards: list[float] = []
        self.raw_image_slot_labels: list[str] = []

        self.current_call_idx = -1
        self.current_image_mask: torch.Tensor | None = None
        self.current_image_slot_labels: list[str] = []
        self.current_raw_slots: list[torch.Tensor] = []
        self.current_vit_inputs: dict[str, torch.Tensor] = {}
        self.current_llm_inputs: dict[str, torch.Tensor] = {}
        self.current_dit_inputs: list[torch.Tensor] = []
        self.current_dit_outputs: list[torch.Tensor] = []
        self.current_final_dit_block_inputs: dict[str, torch.Tensor] = {}
        self.current_action_pred: torch.Tensor | None = None
        self.current_decoded_action: np.ndarray | None = None
        self.action_dim_labels: list[str] = []
        self.current_dit_iteration = -1
        self.current_env_step = 0

        self.prev_raw_slots: list[torch.Tensor] | None = None
        self.prev_vit_inputs: dict[str, torch.Tensor] = {}
        self.prev_llm_inputs: dict[str, torch.Tensor] = {}
        self.prev_final_dit_block_inputs: dict[str, torch.Tensor] = {}
        self.prev_action_pred: torch.Tensor | None = None
        self.prev_decoded_action: np.ndarray | None = None
        self.prev_final_dit_output: torch.Tensor | None = None

        self.vit_layer_names: list[str] = []
        self.llm_layer_names: list[str] = []
        self.dit_block_names: list[str] = []

        self.vit_similarity_mean: dict[str, list[float]] = defaultdict(list)
        self.vit_similarity_slot: dict[str, list[np.ndarray]] = defaultdict(list)
        self.llm_similarity_mean: dict[str, list[float]] = defaultdict(list)
        self.llm_similarity_slot: dict[str, list[np.ndarray]] = defaultdict(list)
        self.dit_block_similarity_mean: dict[str, list[float]] = defaultdict(list)
        self.action_similarity_horizon: list[np.ndarray] = []
        self.action_similarity_dim: list[np.ndarray] = []
        self.decoded_action_similarity_dim: list[np.ndarray] = []
        self.raw_image_similarity_slot: list[np.ndarray] = []
        self.final_dit_output_similarity_action_tokens: list[np.ndarray] = []
        self.dit_iteration_overall_per_call: list[np.ndarray] = []
        self.dit_iteration_action_similarity_accum: np.ndarray | None = None
        self.dit_iteration_action_similarity_count = 0

        self.activation_samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self.activation_sample_count: dict[str, int] = defaultdict(int)
        self.activation_stats_rows: list[dict[str, Any]] = []
        self.weight_rows: list[dict[str, Any]] = []
        self.weight_samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self.weight_sample_count: dict[str, int] = defaultdict(int)

        self.hooks: list[Any] = []
        self._register_hooks()

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _record_activation_stats(self, group: str, layer_name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().float().cpu()
        self.activation_stats_rows.append(
            {
                "group": group,
                "layer": layer_name,
                "call_index": self.current_call_idx,
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "mean_abs": float(tensor.abs().mean().item()),
                "max_abs": float(tensor.abs().max().item()),
            }
        )

    def _append_activation_sample(self, key: str, tensor: torch.Tensor) -> None:
        if self.activation_sample_count[key] >= self.args.max_activation_samples:
            return
        remaining = self.args.max_activation_samples - self.activation_sample_count[key]
        sample = evenly_sample_tensor(tensor, min(self.args.sample_values_per_tensor, remaining))
        if sample.size == 0:
            return
        self.activation_samples[key].append(sample)
        self.activation_sample_count[key] += int(sample.size)

    def _append_weight_sample(self, key: str, tensor: torch.Tensor) -> None:
        if self.weight_sample_count[key] >= self.args.max_activation_samples:
            return
        remaining = self.args.max_activation_samples - self.weight_sample_count[key]
        sample = evenly_sample_tensor(tensor, min(self.args.sample_values_per_tensor, remaining))
        if sample.size == 0:
            return
        self.weight_samples[key].append(sample)
        self.weight_sample_count[key] += int(sample.size)

    def _register_hooks(self) -> None:
        vision_layers = self.model.backbone.model.vision_model.vision_model.encoder.layers
        self.vit_layer_names = [f"vit.layer.{idx}" for idx in range(len(vision_layers))]
        for idx, layer in enumerate(vision_layers):
            name = self.vit_layer_names[idx]

            def make_vit_hook(layer_name: str):
                def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                    hidden_states = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
                    tensor = hidden_states[0].detach().float().cpu()
                    self.current_vit_inputs[layer_name] = tensor
                    self._record_activation_stats("vit", layer_name, tensor)
                    self._append_activation_sample(layer_name, tensor)

                return hook

            self.hooks.append(layer.register_forward_pre_hook(make_vit_hook(name)))

        llm_layers = self.model.backbone.model.language_model.model.layers
        self.llm_layer_names = [f"llm.layer.{idx}" for idx in range(len(llm_layers))]
        for idx, layer in enumerate(llm_layers):
            name = self.llm_layer_names[idx]

            def make_llm_hook(layer_name: str):
                def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                    hidden_states = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
                    tensor = hidden_states.detach().float().cpu()
                    if self.current_image_mask is None:
                        return
                    image_tokens = tensor[0][self.current_image_mask[0].cpu()]
                    self.current_llm_inputs[layer_name] = image_tokens
                    self._record_activation_stats("llm_image_tokens", layer_name, image_tokens)
                    self._append_activation_sample(layer_name, image_tokens)

                return hook

            self.hooks.append(layer.register_forward_pre_hook(make_llm_hook(name)))

        dit_model = self.model.action_head.model

        def dit_model_pre_hook(
            module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            hidden_states = kwargs.get("hidden_states", inputs[0] if inputs else None)
            if hidden_states is None:
                return
            self.current_dit_iteration += 1
            tensor = hidden_states[0].detach().float().cpu()
            self.current_dit_inputs.append(tensor)
            self._record_activation_stats(
                "dit_input", f"dit.iter.{self.current_dit_iteration}.input", tensor
            )
            self._append_activation_sample(f"dit.iter.{self.current_dit_iteration}.input", tensor)

        def dit_model_post_hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor[0].detach().float().cpu()
            self.current_dit_outputs.append(tensor)
            self._record_activation_stats(
                "dit_output", f"dit.iter.{self.current_dit_iteration}.output", tensor
            )
            self._append_activation_sample(f"dit.iter.{self.current_dit_iteration}.output", tensor)

        self.hooks.append(dit_model.register_forward_pre_hook(dit_model_pre_hook, with_kwargs=True))
        self.hooks.append(dit_model.register_forward_hook(dit_model_post_hook, with_kwargs=True))

        dit_blocks = dit_model.transformer_blocks
        self.dit_block_names = [f"dit.block.{idx}" for idx in range(len(dit_blocks))]
        for idx, block in enumerate(dit_blocks):
            name = self.dit_block_names[idx]

            def make_dit_block_hook(layer_name: str):
                def hook(module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                    hidden_states = kwargs.get("hidden_states", inputs[0] if inputs else None)
                    if hidden_states is None:
                        return
                    tensor = hidden_states[0].detach().float().cpu()
                    self._record_activation_stats(
                        "dit_block", f"{layer_name}.iter.{self.current_dit_iteration}", tensor
                    )
                    self._append_activation_sample(layer_name, tensor)
                    if self.current_dit_iteration == self.num_dit_iterations - 1:
                        self.current_final_dit_block_inputs[layer_name] = tensor

                return hook

            self.hooks.append(block.register_forward_pre_hook(make_dit_block_hook(name), with_kwargs=True))

    def start_call(self, env_step: int, nested_obs: dict[str, dict[str, Any]], model_inputs: dict[str, Any]) -> None:
        self.current_call_idx += 1
        self.current_dit_iteration = -1
        self.current_vit_inputs = {}
        self.current_llm_inputs = {}
        self.current_dit_inputs = []
        self.current_dit_outputs = []
        self.current_final_dit_block_inputs = {}
        self.current_action_pred = None
        self.current_decoded_action = None
        self.current_env_step = env_step
        self.current_image_mask = model_inputs["input_ids"] == self.image_token_index
        self.current_image_slot_labels = build_image_slot_labels(nested_obs["video"])
        if not self.raw_image_slot_labels:
            self.raw_image_slot_labels = self.current_image_slot_labels

        self.current_raw_slots = []
        for label in self.current_image_slot_labels:
            t_str, view = label.split(":", 1)
            time_idx = int(t_str[1:])
            arr = nested_obs["video"][view][0, time_idx].astype(np.float32) / 255.0
            self.current_raw_slots.append(torch.from_numpy(arr.reshape(-1)))

    def finish_call(
        self,
        action_pred: torch.Tensor,
        decoded_action: dict[str, np.ndarray],
        reward: float,
    ) -> None:
        self.call_steps.append(self.current_env_step)
        self.call_rewards.append(reward)
        self.current_action_pred = action_pred[0].detach().float().cpu()
        self.current_decoded_action, action_dim_labels = flatten_action_dict(
            decoded_action, self.action_keys
        )
        if not self.action_dim_labels:
            self.action_dim_labels = action_dim_labels

        if self.prev_raw_slots is not None and len(self.prev_raw_slots) == len(self.current_raw_slots):
            sims = [
                cosine_flat(prev_slot, curr_slot)
                for prev_slot, curr_slot in zip(self.prev_raw_slots, self.current_raw_slots)
            ]
            self.raw_image_similarity_slot.append(np.asarray(sims, dtype=np.float32))

        for layer_name, current in self.current_vit_inputs.items():
            previous = self.prev_vit_inputs.get(layer_name)
            if previous is None:
                continue
            sim = cosine_per_row(previous, current)
            self.vit_similarity_mean[layer_name].append(float(sim.mean().item()))
            self.vit_similarity_slot[layer_name].append(
                split_chunk_means(sim, self.current_image_slot_labels)
            )

        for layer_name, current in self.current_llm_inputs.items():
            previous = self.prev_llm_inputs.get(layer_name)
            if previous is None:
                continue
            sim = cosine_per_row(previous, current)
            self.llm_similarity_mean[layer_name].append(float(sim.mean().item()))
            self.llm_similarity_slot[layer_name].append(
                split_chunk_means(sim, self.current_image_slot_labels)
            )

        if self.current_dit_inputs:
            per_iter_scalar = []
            pair_values = []
            for idx in range(1, len(self.current_dit_inputs)):
                prev_tensor = self.current_dit_inputs[idx - 1]
                curr_tensor = self.current_dit_inputs[idx]
                sim = cosine_per_row(prev_tensor[-self.action_horizon :], curr_tensor[-self.action_horizon :])
                pair_values.append(sim.numpy().astype(np.float32))
                per_iter_scalar.append(float(sim.mean().item()))
            if pair_values:
                pair_matrix = np.stack(pair_values, axis=0)
                self.dit_iteration_overall_per_call.append(np.asarray(per_iter_scalar, dtype=np.float32))
                if self.dit_iteration_action_similarity_accum is None:
                    self.dit_iteration_action_similarity_accum = np.zeros_like(pair_matrix)
                self.dit_iteration_action_similarity_accum += pair_matrix
                self.dit_iteration_action_similarity_count += 1

        final_dit_output = self.current_dit_outputs[-1] if self.current_dit_outputs else None
        if self.prev_final_dit_output is not None and final_dit_output is not None:
            sim = cosine_per_row(
                self.prev_final_dit_output[-self.action_horizon :],
                final_dit_output[-self.action_horizon :],
            )
            self.final_dit_output_similarity_action_tokens.append(sim.numpy().astype(np.float32))

        for layer_name, current in self.current_final_dit_block_inputs.items():
            previous = self.prev_final_dit_block_inputs.get(layer_name)
            if previous is None:
                continue
            sim = cosine_per_row(previous[-self.action_horizon :], current[-self.action_horizon :])
            self.dit_block_similarity_mean[layer_name].append(float(sim.mean().item()))

        if self.prev_action_pred is not None:
            prev_action = self.prev_action_pred
            curr_action = self.current_action_pred
            self.action_similarity_horizon.append(
                F.cosine_similarity(prev_action.float(), curr_action.float(), dim=-1, eps=1e-8)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            self.action_similarity_dim.append(
                F.cosine_similarity(
                    prev_action.float().transpose(0, 1),
                    curr_action.float().transpose(0, 1),
                    dim=-1,
                    eps=1e-8,
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if self.prev_decoded_action is not None and self.current_decoded_action is not None:
            prev_action = torch.from_numpy(self.prev_decoded_action)
            curr_action = torch.from_numpy(self.current_decoded_action)
            self.decoded_action_similarity_dim.append(
                F.cosine_similarity(
                    prev_action.float().transpose(0, 1),
                    curr_action.float().transpose(0, 1),
                    dim=-1,
                    eps=1e-8,
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        self.prev_raw_slots = [slot.clone() for slot in self.current_raw_slots]
        self.prev_vit_inputs = {name: tensor.clone() for name, tensor in self.current_vit_inputs.items()}
        self.prev_llm_inputs = {name: tensor.clone() for name, tensor in self.current_llm_inputs.items()}
        self.prev_final_dit_block_inputs = {
            name: tensor.clone() for name, tensor in self.current_final_dit_block_inputs.items()
        }
        self.prev_action_pred = self.current_action_pred.clone()
        self.prev_decoded_action = self.current_decoded_action.copy()
        if final_dit_output is not None:
            self.prev_final_dit_output = final_dit_output.clone()

    def analyze_weights(self) -> None:
        for name, param in self.model.named_parameters():
            tensor = param.detach().float().cpu()
            group = self._classify_weight_group(name)
            self.weight_rows.append(
                {
                    "name": name,
                    "group": group,
                    "numel": int(tensor.numel()),
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std(unbiased=False).item()),
                    "mean_abs": float(tensor.abs().mean().item()),
                    "max_abs": float(tensor.abs().max().item()),
                    "l2_norm": float(torch.linalg.vector_norm(tensor).item()),
                }
            )
            self._append_weight_sample(group, tensor)

    def _classify_weight_group(self, name: str) -> str:
        if name.startswith("backbone.model.vision_model"):
            return "vision_vit"
        if name.startswith("backbone.model.mlp1"):
            return "vision_projector"
        if name.startswith("backbone.model.language_model"):
            return "vlm_llm"
        if name.startswith("action_head.model"):
            return "dit"
        if name.startswith("action_head.state_encoder"):
            return "state_encoder"
        if name.startswith("action_head.action_encoder"):
            return "action_encoder"
        if name.startswith("action_head.action_decoder"):
            return "action_decoder"
        if name.startswith("action_head.vlln"):
            return "vlln"
        if name.startswith("action_head.position_embedding"):
            return "action_pos_embed"
        return "other"

    def _merge_sample_lists(self, sample_dict: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        merged = {}
        for key, values in sample_dict.items():
            merged[key] = np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        return merged

    def _build_layer_matrix(
        self,
        layer_names: list[str],
        series_map: dict[str, list[float]],
    ) -> np.ndarray:
        if not layer_names:
            return np.empty((0, 0), dtype=np.float32)
        length = max((len(series_map.get(name, [])) for name in layer_names), default=0)
        matrix = np.full((len(layer_names), length), np.nan, dtype=np.float32)
        for row_idx, layer_name in enumerate(layer_names):
            values = np.asarray(series_map.get(layer_name, []), dtype=np.float32)
            matrix[row_idx, : values.shape[0]] = values
        return matrix

    def save_outputs(self, meta: dict[str, Any]) -> None:
        ensure_dir(self.output_dir)

        self.analyze_weights()

        vit_matrix = self._build_layer_matrix(self.vit_layer_names, self.vit_similarity_mean)
        llm_matrix = self._build_layer_matrix(self.llm_layer_names, self.llm_similarity_mean)
        dit_block_matrix = self._build_layer_matrix(self.dit_block_names, self.dit_block_similarity_mean)

        raw_image_matrix = (
            np.stack(self.raw_image_similarity_slot, axis=0)
            if self.raw_image_similarity_slot
            else np.empty((0, len(self.raw_image_slot_labels)), dtype=np.float32)
        )
        action_horizon_matrix = (
            np.stack(self.action_similarity_horizon, axis=0)
            if self.action_similarity_horizon
            else np.empty((0, self.action_horizon), dtype=np.float32)
        )
        action_dim_matrix = (
            np.stack(self.action_similarity_dim, axis=0)
            if self.action_similarity_dim
            else np.empty((0, 0), dtype=np.float32)
        )
        decoded_action_dim_matrix = (
            np.stack(self.decoded_action_similarity_dim, axis=0)
            if self.decoded_action_similarity_dim
            else np.empty((0, 0), dtype=np.float32)
        )
        final_dit_output_matrix = (
            np.stack(self.final_dit_output_similarity_action_tokens, axis=0)
            if self.final_dit_output_similarity_action_tokens
            else np.empty((0, self.action_horizon), dtype=np.float32)
        )
        dit_iteration_overall = (
            np.stack(self.dit_iteration_overall_per_call, axis=0)
            if self.dit_iteration_overall_per_call
            else np.empty((0, max(self.num_dit_iterations - 1, 0)), dtype=np.float32)
        )
        dit_iteration_mean = (
            self.dit_iteration_action_similarity_accum / max(self.dit_iteration_action_similarity_count, 1)
            if self.dit_iteration_action_similarity_accum is not None
            else np.empty((0, self.action_horizon), dtype=np.float32)
        )

        np.savez_compressed(
            self.output_dir / "metrics.npz",
            policy_call_steps=np.asarray(self.call_steps, dtype=np.int32),
            policy_call_rewards=np.asarray(self.call_rewards, dtype=np.float32),
            raw_image_similarity=raw_image_matrix,
            vit_similarity_mean=vit_matrix,
            llm_similarity_mean=llm_matrix,
            dit_block_similarity_mean=dit_block_matrix,
            dit_iteration_action_similarity_mean=dit_iteration_mean,
            dit_iteration_overall=dit_iteration_overall,
            action_similarity_horizon=action_horizon_matrix,
            action_similarity_dim=action_dim_matrix,
            decoded_action_similarity_dim=decoded_action_dim_matrix,
            final_dit_output_similarity_action_tokens=final_dit_output_matrix,
        )

        if raw_image_matrix.size:
            save_heatmap(
                self.output_dir / "raw_image_similarity.png",
                raw_image_matrix,
                "Raw Image Similarity Between Consecutive Policy Calls",
                "Image Slot",
                "Policy Call Pair",
                xlabels=self.raw_image_slot_labels,
                cmap="magma",
                vmin=-1.0,
                vmax=1.0,
            )

        if vit_matrix.size:
            save_heatmap(
                self.output_dir / "vit_layer_similarity.png",
                vit_matrix,
                "ViT Input Token Similarity Between Consecutive Policy Calls",
                "Policy Call Pair",
                "ViT Layer",
                ylabels=self.vit_layer_names,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if llm_matrix.size:
            save_heatmap(
                self.output_dir / "llm_image_token_similarity.png",
                llm_matrix,
                "VLM Image-Token Similarity Between Consecutive Policy Calls",
                "Policy Call Pair",
                "LLM Layer",
                ylabels=self.llm_layer_names,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        detailed_vit_layers = choose_representative_indices(len(self.vit_layer_names))
        for idx in detailed_vit_layers:
            layer_name = self.vit_layer_names[idx]
            slot_values = self.vit_similarity_slot.get(layer_name, [])
            if not slot_values:
                continue
            matrix = np.stack(slot_values, axis=0)
            save_heatmap(
                self.output_dir / f"{layer_name.replace('.', '_')}_slot_similarity.png",
                matrix,
                f"{layer_name} Slot Similarity",
                "Image Slot",
                "Policy Call Pair",
                xlabels=self.raw_image_slot_labels,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        detailed_llm_layers = choose_representative_indices(len(self.llm_layer_names))
        for idx in detailed_llm_layers:
            layer_name = self.llm_layer_names[idx]
            slot_values = self.llm_similarity_slot.get(layer_name, [])
            if not slot_values:
                continue
            matrix = np.stack(slot_values, axis=0)
            save_heatmap(
                self.output_dir / f"{layer_name.replace('.', '_')}_slot_similarity.png",
                matrix,
                f"{layer_name} Image Token Slot Similarity",
                "Image Slot",
                "Policy Call Pair",
                xlabels=self.raw_image_slot_labels,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if dit_iteration_mean.size:
            save_heatmap(
                self.output_dir / "dit_iteration_action_similarity_mean.png",
                dit_iteration_mean,
                "Mean DiT Input Similarity Across Consecutive Iterations",
                "Action Token Position",
                "Iteration Pair",
                xlabels=[f"a{i}" for i in range(dit_iteration_mean.shape[1])],
                ylabels=[f"{i}->{i + 1}" for i in range(dit_iteration_mean.shape[0])],
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if dit_iteration_overall.size:
            save_heatmap(
                self.output_dir / "dit_iteration_similarity_by_call.png",
                dit_iteration_overall,
                "DiT Input Similarity By Policy Call",
                "Iteration Pair",
                "Policy Call",
                xlabels=[f"{i}->{i + 1}" for i in range(dit_iteration_overall.shape[1])],
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if dit_block_matrix.size:
            save_heatmap(
                self.output_dir / "dit_block_similarity.png",
                dit_block_matrix,
                "Final-Iteration DiT Block Input Similarity Between Consecutive Policy Calls",
                "Policy Call Pair",
                "DiT Block",
                ylabels=self.dit_block_names,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if action_horizon_matrix.size:
            save_heatmap(
                self.output_dir / "action_similarity_horizon.png",
                action_horizon_matrix,
                "Normalized Action Similarity By Horizon",
                "Action Horizon Position",
                "Policy Call Pair",
                xlabels=[f"h{i}" for i in range(action_horizon_matrix.shape[1])],
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if action_dim_matrix.size:
            save_heatmap(
                self.output_dir / "action_similarity_dimension.png",
                action_dim_matrix,
                "Normalized Action Similarity By Dimension",
                "Action Dimension",
                "Policy Call Pair",
                xlabels=self.action_dim_labels if len(self.action_dim_labels) == action_dim_matrix.shape[1] else None,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if decoded_action_dim_matrix.size:
            save_heatmap(
                self.output_dir / "decoded_action_similarity_dimension.png",
                decoded_action_dim_matrix,
                "Decoded Action Similarity By Dimension",
                "Decoded Action Dimension",
                "Policy Call Pair",
                xlabels=self.action_dim_labels
                if len(self.action_dim_labels) == decoded_action_dim_matrix.shape[1]
                else None,
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        if final_dit_output_matrix.size:
            save_heatmap(
                self.output_dir / "final_dit_output_action_similarity.png",
                final_dit_output_matrix,
                "Final DiT Output Similarity On Action Tokens",
                "Action Token Position",
                "Policy Call Pair",
                xlabels=[f"a{i}" for i in range(final_dit_output_matrix.shape[1])],
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
            )

        save_line_plot(
            self.output_dir / "reward_timeline.png",
            np.asarray(self.call_steps, dtype=np.int32),
            {"reward": np.asarray(self.call_rewards, dtype=np.float32)},
            "Reward Per Policy Call",
            "Env Step",
            "Reward",
        )

        merged_activation_samples = self._merge_sample_lists(self.activation_samples)
        sample_hist_keys = {}
        for idx in choose_representative_indices(len(self.vit_layer_names)):
            key = self.vit_layer_names[idx]
            if key in merged_activation_samples:
                sample_hist_keys[key] = merged_activation_samples[key]
        for idx in choose_representative_indices(len(self.llm_layer_names)):
            key = self.llm_layer_names[idx]
            if key in merged_activation_samples:
                sample_hist_keys[key] = merged_activation_samples[key]
        for idx in choose_representative_indices(self.num_dit_iterations):
            key = f"dit.iter.{idx}.input"
            if key in merged_activation_samples:
                sample_hist_keys[key] = merged_activation_samples[key]
        save_histograms(
            self.output_dir / "activation_histograms.png",
            sample_hist_keys,
            "Activation Distribution Samples",
            "Activation Value",
        )

        merged_weight_samples = self._merge_sample_lists(self.weight_samples)
        save_boxplot(
            self.output_dir / "weight_group_boxplot.png",
            merged_weight_samples,
            "Weight Distribution By Model Part",
            "Weight Value",
        )

        save_csv(self.output_dir / "activation_stats.csv", self.activation_stats_rows)
        save_csv(self.output_dir / "weight_stats.csv", self.weight_rows)

        summary = {
            **meta,
            "num_policy_calls": self.current_call_idx + 1,
            "num_policy_call_pairs": max(self.current_call_idx, 0),
            "output_dir": str(self.output_dir),
            "raw_image_slot_labels": self.raw_image_slot_labels,
            "action_dim_labels": self.action_dim_labels,
            "vit_layers": self.vit_layer_names,
            "llm_layers": self.llm_layer_names,
            "dit_blocks": self.dit_block_names,
            "reward_mean": float(np.mean(self.call_rewards)) if self.call_rewards else 0.0,
            "reward_max": float(np.max(self.call_rewards)) if self.call_rewards else 0.0,
            "raw_image_similarity_mean": float(np.nanmean(raw_image_matrix))
            if raw_image_matrix.size
            else None,
            "vit_similarity_mean": float(np.nanmean(vit_matrix)) if vit_matrix.size else None,
            "llm_similarity_mean": float(np.nanmean(llm_matrix)) if llm_matrix.size else None,
            "dit_iteration_similarity_mean": float(np.nanmean(dit_iteration_mean))
            if dit_iteration_mean.size
            else None,
            "dit_block_similarity_mean": float(np.nanmean(dit_block_matrix))
            if dit_block_matrix.size
            else None,
            "action_similarity_horizon_mean": float(np.nanmean(action_horizon_matrix))
            if action_horizon_matrix.size
            else None,
            "action_similarity_dim_mean": float(np.nanmean(action_dim_matrix))
            if action_dim_matrix.size
            else None,
            "decoded_action_similarity_dim_mean": float(np.nanmean(decoded_action_dim_matrix))
            if decoded_action_dim_matrix.size
            else None,
            "final_dit_output_similarity_mean": float(np.nanmean(final_dit_output_matrix))
            if final_dit_output_matrix.size
            else None,
        }
        save_json(self.output_dir / "summary.json", summary)


class InstrumentedPolicyRunner:
    def __init__(self, policy: Gr00tPolicy, analyzer: SimilarityAnalyzer):
        self.policy = policy
        self.analyzer = analyzer
        self.processor: BaseProcessor = policy.processor
        self.collate_fn = policy.collate_fn
        self.modality_configs = policy.modality_configs
        self.language_key = policy.language_key

    def _flat_to_nested_observation(self, observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        nested = {"video": {}, "state": {}, "language": {}}
        for modality in ["video", "state", "language"]:
            for key in self.modality_configs[modality].modality_keys:
                if modality == "language":
                    nested[modality][key] = [[str(item)] for item in observation[key]]
                else:
                    nested[modality][key] = observation[f"{modality}.{key}"]
        return nested

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=EmbodimentTag.LIBERO_PANDA,
        )

    def get_action(
        self, observation: dict[str, Any], env_step: int
    ) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, np.ndarray]]:
        nested_obs = self._flat_to_nested_observation(observation)
        if self.policy.strict:
            self.policy.check_observation(nested_obs)

        unbatched_observations = self.policy._unbatch_observation(nested_obs)
        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        model_inputs = collated_inputs["inputs"]

        self.analyzer.start_call(env_step, nested_obs, model_inputs)
        with torch.inference_mode():
            model_pred = self.policy.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        batched_states = {}
        for state_key in self.modality_configs["state"].modality_keys:
            batched_states[state_key] = np.stack([state[state_key] for state in states], axis=0)
        decoded_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), EmbodimentTag.LIBERO_PANDA, batched_states
        )
        flat_action = {f"action.{key}": value.astype(np.float32) for key, value in decoded_action.items()}
        return flat_action, normalized_action, decoded_action


def build_env(env_name: str, max_episode_steps: int, n_action_steps: int, save_video: bool) -> gym.vector.SyncVectorEnv:
    register_libero_envs()
    video_dir = None
    if save_video:
        video_dir = str(Path("/tmp") / f"libero_analysis_{env_name.split('/')[-1]}")
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=video_dir, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    env_fns = [
        lambda: create_eval_env(
            env_name=env_name,
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=wrapper_configs,
        )
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def extract_success(env_infos: dict[str, Any], current_success: bool) -> bool:
    if "success" in env_infos:
        env_success = env_infos["success"][0]
        if isinstance(env_success, list):
            current_success |= any(env_success)
        elif isinstance(env_success, np.ndarray):
            current_success |= bool(np.any(env_success))
        else:
            current_success |= bool(env_success)
    if "final_info" in env_infos and env_infos["final_info"][0] is not None:
        env_success = env_infos["final_info"][0]["success"]
        if isinstance(env_success, list):
            current_success |= any(env_success)
        elif isinstance(env_success, np.ndarray):
            current_success |= bool(np.any(env_success))
        else:
            current_success |= bool(env_success)
    return current_success


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )
    analyzer = SimilarityAnalyzer(policy=policy, output_dir=output_dir, args=args)
    runner = InstrumentedPolicyRunner(policy=policy, analyzer=analyzer)
    env = build_env(
        env_name=args.env_name,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        save_video=args.save_video,
    )

    try:
        observations, _ = env.reset(seed=args.seed)
        policy.reset()
        current_success = False
        env_step = 0
        total_reward = 0.0
        done = False

        while not done:
            if args.max_policy_calls and analyzer.current_call_idx + 1 >= args.max_policy_calls:
                break

            action, normalized_action, decoded_action = runner.get_action(observations, env_step)
            next_obs, rewards, terminations, truncations, env_infos = env.step(action)
            reward = float(rewards[0])
            total_reward += reward
            current_success = extract_success(env_infos, current_success)
            analyzer.finish_call(normalized_action, decoded_action, reward)

            observations = next_obs
            done = bool(terminations[0] or truncations[0])
            env_step += 1

        meta = {
            "env_name": args.env_name,
            "model_path": args.model_path,
            "device": args.device,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "n_action_steps": args.n_action_steps,
            "total_reward": total_reward,
            "success": bool(current_success),
            "terminated": bool(done),
            "policy_calls": analyzer.current_call_idx + 1,
        }
        analyzer.save_outputs(meta)
    finally:
        analyzer.close()
        env.close()

    print(f"Saved analysis to: {output_dir}")
    print(f"Success: {meta['success']}")
    print(f"Policy calls: {meta['policy_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
