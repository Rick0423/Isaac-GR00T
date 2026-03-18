from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.policy.policy import PolicyWrapper


def _safe_float(value: torch.Tensor | float | np.ndarray) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _activation_stats(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    return {
        "mean": _safe_float(tensor.mean()),
        "std": _safe_float(tensor.std(unbiased=False)),
        "abs_mean": _safe_float(tensor.abs().mean()),
        "norm_mean": _safe_float(torch.linalg.vector_norm(tensor, dim=-1).mean()),
        "min": _safe_float(tensor.min()),
        "max": _safe_float(tensor.max()),
    }


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    flat = vectors.reshape(vectors.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = flat / norms
    return normalized @ normalized.T


def _mean_off_diagonal(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def _consecutive_cosines(vectors: list[np.ndarray]) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(
        [_cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)],
        dtype=np.float32,
    )


def _evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=count, dtype=np.int64)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _get_nested_attr(root: Any, candidates: list[str]) -> Any:
    for candidate in candidates:
        current = root
        ok = True
        for part in candidate.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok:
            return current
    raise AttributeError(f"Unable to resolve any of: {candidates}")


def _parse_index_spec(spec: str, total: int) -> list[int]:
    if total <= 0:
        return []
    if not spec or spec.lower() == "auto":
        return sorted({0, total // 2, total - 1})
    if spec.lower() == "all":
        return list(range(total))
    indices = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "first":
            indices.append(0)
        elif token == "middle":
            indices.append(total // 2)
        elif token == "last":
            indices.append(total - 1)
        else:
            index = int(token)
            if index < 0:
                index = total + index
            if not 0 <= index < total:
                raise ValueError(f"Index {token} is out of range for total={total}")
            indices.append(index)
    return sorted(set(indices))


@dataclass
class AnalysisConfig:
    output_dir: str = "/tmp/gr00t_inference_analysis"
    token_sample_size: int = 12
    raw_image_sample_size: int = 4096
    detail_layer_indices: str = "auto"
    detail_request_indices: str = "auto"
    dit_block_indices: str = "auto"
    weight_sample_size: int = 200000
    plot_formats: str = "png"
    auto_dump_on_exit: bool = True
    auto_dump_on_reset: bool = False
    save_raw_tensors: bool = True


@dataclass
class ImageSpatialRecord:
    label: str
    token_count: int
    sampled_similarity: np.ndarray
    sampled_tokens: np.ndarray
    stats: dict[str, float]
    mean_pairwise_cosine: float


@dataclass
class FamilyLayerRecord:
    pooled: np.ndarray
    stats: dict[str, float]
    images: dict[str, ImageSpatialRecord] = field(default_factory=dict)


@dataclass
class RequestRecord:
    request_index: int
    image_labels: list[str] = field(default_factory=list)
    raw_image_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    vision: dict[str, FamilyLayerRecord] = field(default_factory=dict)
    vlm: dict[str, FamilyLayerRecord] = field(default_factory=dict)
    dit_model_inputs: list[np.ndarray] = field(default_factory=list)
    dit_model_input_stats: list[dict[str, float]] = field(default_factory=list)
    dit_blocks: dict[str, list[np.ndarray | None]] = field(default_factory=dict)
    dit_block_stats: dict[str, list[dict[str, float] | None]] = field(default_factory=dict)
    pred_velocities: list[np.ndarray] = field(default_factory=list)
    action_pred: np.ndarray | None = None


class InferenceAnalyzer:
    def __init__(self, model: torch.nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.output_root = Path(config.output_dir).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.plot_formats = [fmt.strip() for fmt in config.plot_formats.split(",") if fmt.strip()]

        self.records: list[RequestRecord] = []
        self.request_counter = 0
        self.dump_counter = 0
        self.current_record: RequestRecord | None = None
        self.current_diffusion_iteration: int | None = None
        self.current_vlm_image_mask: torch.Tensor | None = None
        self.current_image_slices: list[tuple[str, slice]] = []
        self.current_image_labels: list[str] = []

        self.hooks: list[Any] = []
        self.vision_layers = self._resolve_vision_layers()
        self.vlm_layers = self._resolve_vlm_layers()
        self.dit_blocks = self._resolve_dit_blocks()
        self.selected_dit_block_names = [
            name
            for idx, name in enumerate(self.dit_blocks.keys())
            if idx in _parse_index_spec(self.config.dit_block_indices, len(self.dit_blocks))
        ]
        self._install_hooks()

    def _resolve_vision_layers(self) -> dict[str, torch.nn.Module]:
        tower = _get_nested_attr(
            self.model.backbone.model.vision_model,
            ["vision_model.encoder.layers", "encoder.layers", "model.encoder.layers"],
        )
        return {f"vision.layer_{idx:02d}": module for idx, module in enumerate(tower)}

    def _resolve_vlm_layers(self) -> dict[str, torch.nn.Module]:
        layers = _get_nested_attr(
            self.model.backbone.model.language_model,
            ["model.layers", "model.model.layers", "base_model.model.model.layers"],
        )
        return {f"vlm.layer_{idx:02d}": module for idx, module in enumerate(layers)}

    def _resolve_dit_blocks(self) -> dict[str, torch.nn.Module]:
        return {
            f"dit.block_{idx:02d}": module
            for idx, module in enumerate(self.model.action_head.model.transformer_blocks)
        }

    def _install_hooks(self):
        for layer_name, module in self.vision_layers.items():
            self.hooks.append(
                module.register_forward_pre_hook(self._make_family_hook("vision", layer_name))
            )
        for layer_name, module in self.vlm_layers.items():
            self.hooks.append(
                module.register_forward_pre_hook(self._make_family_hook("vlm", layer_name))
            )
        for layer_name, module in self.dit_blocks.items():
            self.hooks.append(module.register_forward_pre_hook(self._make_dit_hook(layer_name)))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset(self) -> dict[str, Any]:
        dump_result = None
        if self.config.auto_dump_on_reset and self.records:
            dump_result = self.dump()
        self.records = []
        self.request_counter = 0
        self.current_record = None
        self.current_diffusion_iteration = None
        self.current_vlm_image_mask = None
        self.current_image_slices = []
        self.current_image_labels = []
        payload = {"status": "ok", "message": "analysis buffers cleared"}
        if dump_result is not None:
            payload["dump_result"] = dump_result
        return payload

    def status(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "recorded_requests": len(self.records),
            "vision_layers": list(self.vision_layers.keys()),
            "vlm_layers": list(self.vlm_layers.keys()),
            "selected_dit_blocks": self.selected_dit_block_names,
            "dump_counter": self.dump_counter,
        }

    def _extract_image_token_counts(self, backbone_inputs: dict[str, Any]) -> list[int]:
        patch_size = self.model.backbone.model.vision_model.vision_model.embeddings.patch_size
        counts = []
        for batch_tensor in backbone_inputs["pixel_values"]:
            if batch_tensor.ndim != 4:
                raise ValueError(f"Unexpected pixel_values tensor shape: {tuple(batch_tensor.shape)}")
            _, _, height, width = batch_tensor.shape
            token_count = (height // patch_size) * (width // patch_size)
            counts.extend([token_count] * batch_tensor.shape[0])
        return counts

    def begin_request(
        self,
        backbone_inputs: dict[str, Any],
        image_labels: list[str],
        raw_image_vectors: dict[str, np.ndarray],
    ):
        image_mask = backbone_inputs["input_ids"] == self.model.backbone.model.image_token_index
        if image_mask.shape[0] != 1:
            raise ValueError(
                "Analysis mode requires batch size 1. Please run the client with n_envs=1."
            )
        token_counts = self._extract_image_token_counts(backbone_inputs)
        if len(token_counts) != len(image_labels):
            raise ValueError(
                f"Image count mismatch: token_counts={len(token_counts)} image_labels={len(image_labels)}"
            )
        self.current_image_slices = []
        offset = 0
        for label, count in zip(image_labels, token_counts):
            self.current_image_slices.append((label, slice(offset, offset + count)))
            offset += count
        self.current_vlm_image_mask = image_mask[0]
        self.current_image_labels = list(image_labels)
        self.current_record = RequestRecord(
            request_index=self.request_counter,
            image_labels=list(image_labels),
            raw_image_vectors={k: np.asarray(v, dtype=np.float32) for k, v in raw_image_vectors.items()},
        )
        self.request_counter += 1

    def end_request(self):
        if self.current_record is not None:
            self.records.append(self.current_record)
        self.current_record = None
        self.current_diffusion_iteration = None
        self.current_vlm_image_mask = None
        self.current_image_slices = []
        self.current_image_labels = []

    def _make_family_hook(self, family: str, layer_name: str):
        def hook(_module, args):
            if self.current_record is None:
                return
            hidden_states = self._extract_first_tensor_arg(args)
            if hidden_states is None or hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                return
            if family == "vision":
                record = self._capture_family(hidden_states[0], layer_name, family)
                self.current_record.vision[layer_name] = record
            else:
                if self.current_vlm_image_mask is None:
                    return
                image_tokens = hidden_states[0][self.current_vlm_image_mask]
                if image_tokens.numel() == 0:
                    return
                record = self._capture_family(image_tokens, layer_name, family)
                self.current_record.vlm[layer_name] = record

        return hook

    def _make_dit_hook(self, layer_name: str):
        def hook(_module, args):
            if (
                self.current_record is None
                or self.current_diffusion_iteration is None
                or layer_name not in self.selected_dit_block_names
            ):
                return
            hidden_states = self._extract_first_tensor_arg(args)
            if hidden_states is None or hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                return
            tensor = hidden_states[0].detach().float()
            if layer_name not in self.current_record.dit_blocks:
                total_iters = self.model.action_head.num_inference_timesteps
                self.current_record.dit_blocks[layer_name] = [None] * total_iters
                self.current_record.dit_block_stats[layer_name] = [None] * total_iters
            self.current_record.dit_blocks[layer_name][self.current_diffusion_iteration] = (
                tensor.to(dtype=torch.float16).cpu().numpy()
            )
            self.current_record.dit_block_stats[layer_name][self.current_diffusion_iteration] = (
                _activation_stats(tensor)
            )

        return hook

    def _extract_first_tensor_arg(self, args: tuple[Any, ...]) -> torch.Tensor | None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg
        return None

    def _capture_family(
        self,
        tokens: torch.Tensor,
        layer_name: str,
        family: str,
    ) -> FamilyLayerRecord:
        tokens = tokens.detach().float()
        images: dict[str, ImageSpatialRecord] = {}
        for label, token_slice in self.current_image_slices:
            image_tokens = tokens[token_slice]
            if image_tokens.numel() == 0:
                continue
            token_indices = _evenly_spaced_indices(
                image_tokens.shape[0], self.config.token_sample_size
            )
            sampled_tokens = image_tokens[token_indices]
            sampled_similarity = _pairwise_cosine_matrix(sampled_tokens.cpu().numpy())
            images[label] = ImageSpatialRecord(
                label=label,
                token_count=int(image_tokens.shape[0]),
                sampled_similarity=sampled_similarity.astype(np.float32),
                sampled_tokens=sampled_tokens.to(dtype=torch.float16).cpu().numpy(),
                stats=_activation_stats(image_tokens),
                mean_pairwise_cosine=_mean_off_diagonal(sampled_similarity),
            )
        return FamilyLayerRecord(
            pooled=tokens.mean(dim=0).to(dtype=torch.float16).cpu().numpy(),
            stats=_activation_stats(tokens),
            images=images,
        )

    @torch.no_grad()
    def run_model(
        self,
        collated_inputs: dict[str, Any],
        image_labels: list[str],
        raw_image_vectors: dict[str, np.ndarray],
    ) -> torch.Tensor:
        if "inputs" in collated_inputs:
            collated_inputs = collated_inputs["inputs"]
        backbone_inputs, action_inputs = self.model.prepare_input(collated_inputs)
        self.begin_request(backbone_inputs, image_labels=image_labels, raw_image_vectors=raw_image_vectors)
        try:
            backbone_outputs = self.model.backbone(backbone_inputs)
            action_pred = self._run_action_head(self.model.action_head, backbone_outputs, action_inputs)
            self.current_record.action_pred = action_pred[0].detach().float().cpu().numpy()
            return action_pred
        finally:
            self.end_request()

    def _run_action_head(
        self,
        action_head: torch.nn.Module,
        backbone_output: Any,
        action_input: Any,
    ) -> torch.Tensor:
        features = action_head._encode_features(backbone_output, action_input)
        vl_embeds = features.backbone_features
        state_features = features.state_features
        embodiment_id = action_input.embodiment_id

        batch_size = vl_embeds.shape[0]
        if batch_size != 1:
            raise ValueError(
                "Analysis mode requires batch size 1 for DiT tracing. Please run the client with n_envs=1."
            )
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        dt = 1.0 / action_head.num_inference_timesteps

        for iteration in range(action_head.num_inference_timesteps):
            t_cont = iteration / float(action_head.num_inference_timesteps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)
            timestep_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = action_head.action_encoder(actions, timestep_tensor, embodiment_id)
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            self.current_record.dit_model_inputs.append(
                sa_embs[0].detach().float().to(dtype=torch.float16).cpu().numpy()
            )
            self.current_record.dit_model_input_stats.append(_activation_stats(sa_embs[0]))
            self.current_diffusion_iteration = iteration

            if action_head.config.use_alternate_vl_dit:
                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timestep_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timestep_tensor,
                )
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            self.current_record.pred_velocities.append(
                pred_velocity[0].detach().float().cpu().numpy()
            )
            actions = actions + dt * pred_velocity

        self.current_diffusion_iteration = None
        return actions

    def dump(self) -> dict[str, Any]:
        dump_dir = self.output_root / f"run_{self.dump_counter:03d}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        summary = self._build_summary()
        self._write_json(dump_dir / "summary.json", summary)
        if self.config.save_raw_tensors and self.records:
            torch.save(self._build_raw_payload(summary), dump_dir / "raw_traces.pt")
        self._plot_all(dump_dir, summary)
        self.dump_counter += 1
        return {
            "status": "ok",
            "dump_dir": str(dump_dir),
            "summary_path": str(dump_dir / "summary.json"),
        }

    def _build_summary(self) -> dict[str, Any]:
        action_preds = [record.action_pred for record in self.records if record.action_pred is not None]
        consecutive_action_similarity = _consecutive_cosines([item.reshape(-1) for item in action_preds])
        first_labels = self.records[0].image_labels if self.records else []
        return {
            "analysis_config": asdict(self.config),
            "recorded_requests": len(self.records),
            "image_labels": first_labels,
            "vision_layers": list(self.vision_layers.keys()),
            "vlm_layers": list(self.vlm_layers.keys()),
            "selected_dit_blocks": self.selected_dit_block_names,
            "consecutive_action_similarity_mean": (
                float(consecutive_action_similarity.mean())
                if consecutive_action_similarity.size
                else 0.0
            ),
            "weights": self._summarize_weights(),
        }

    def _build_raw_payload(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {"summary": summary, "records": self.records}

    def _write_json(self, path: Path, payload: dict[str, Any]):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _plot_all(self, dump_dir: Path, summary: dict[str, Any]):
        self._plot_family(
            dump_dir / "vision",
            "Vision Transformer Block Inputs",
            list(self.vision_layers.keys()),
            [record.vision for record in self.records],
            summary.get("image_labels", []),
        )
        self._plot_family(
            dump_dir / "vlm",
            "VLM Image Token Inputs",
            list(self.vlm_layers.keys()),
            [record.vlm for record in self.records],
            summary.get("image_labels", []),
        )
        self._plot_dit(dump_dir / "dit")
        self._plot_requestwise_similarity(dump_dir / "requestwise")
        self._plot_weights(dump_dir / "weights", summary["weights"])

    def _plot_family(
        self,
        output_dir: Path,
        title_prefix: str,
        layer_names: list[str],
        family_records: list[dict[str, FamilyLayerRecord]],
        image_labels: list[str],
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        if not family_records or not layer_names or not image_labels:
            return

        mean_spatial = np.full((len(layer_names), len(image_labels)), np.nan, dtype=np.float32)
        mean_std = np.full((len(layer_names), len(image_labels)), np.nan, dtype=np.float32)
        consecutive_rows = []

        for layer_idx, layer_name in enumerate(layer_names):
            pooled_vectors = []
            for request in family_records:
                record = request.get(layer_name)
                if record is None:
                    continue
                pooled_vectors.append(np.asarray(record.pooled, dtype=np.float32).reshape(-1))

            consecutive_rows.append(_consecutive_cosines(pooled_vectors))

            for image_idx, image_label in enumerate(image_labels):
                similarities = []
                std_values = []
                for request in family_records:
                    record = request.get(layer_name)
                    if record is None:
                        continue
                    image_record = record.images.get(image_label)
                    if image_record is None:
                        continue
                    similarities.append(image_record.mean_pairwise_cosine)
                    std_values.append(image_record.stats["std"])
                if similarities:
                    mean_spatial[layer_idx, image_idx] = float(np.mean(similarities))
                if std_values:
                    mean_std[layer_idx, image_idx] = float(np.mean(std_values))

        self._save_heatmap(
            output_dir / "mean_spatial_similarity_by_layer",
            mean_spatial,
            f"{title_prefix}: Mean Spatial Token Cosine",
            xlabel="input_image",
            ylabel="layer",
            xticklabels=image_labels,
            yticklabels=layer_names,
        )
        self._save_heatmap(
            output_dir / "mean_activation_std_by_layer",
            mean_std,
            f"{title_prefix}: Mean Activation Std",
            xlabel="input_image",
            ylabel="layer",
            xticklabels=image_labels,
            yticklabels=layer_names,
        )
        self._save_ragged_heatmap(
            output_dir / "consecutive_request_similarity_by_layer",
            consecutive_rows,
            layer_names,
            f"{title_prefix}: Consecutive Request Pooled Cosine",
            value_label="cosine",
        )

        detail_layers = _parse_index_spec(self.config.detail_layer_indices, len(layer_names))
        detail_requests = _parse_index_spec(self.config.detail_request_indices, len(family_records))
        for request_idx in detail_requests:
            detail_dir = output_dir / f"request_{request_idx:03d}"
            detail_dir.mkdir(parents=True, exist_ok=True)
            for layer_idx in detail_layers:
                layer_name = layer_names[layer_idx]
                record = family_records[request_idx].get(layer_name)
                if record is None:
                    continue
                for image_label in image_labels:
                    image_record = record.images.get(image_label)
                    if image_record is None:
                        continue
                    self._save_heatmap(
                        detail_dir / f"{_sanitize_name(layer_name)}_{_sanitize_name(image_label)}_spatial_similarity",
                        image_record.sampled_similarity,
                        f"{title_prefix}: {layer_name} {image_label} Spatial Cosine",
                        xlabel="sampled_token",
                        ylabel="sampled_token",
                    )

        for layer_idx in detail_layers:
            layer_name = layer_names[layer_idx]
            sampled_values = []
            for request in family_records:
                record = request.get(layer_name)
                if record is None:
                    continue
                for image_record in record.images.values():
                    sampled_values.append(image_record.sampled_tokens.reshape(-1))
            if sampled_values:
                self._save_histogram(
                    output_dir / f"{_sanitize_name(layer_name)}_activation_hist",
                    np.concatenate(sampled_values, axis=0),
                    f"{title_prefix}: {layer_name} Sampled Activation Distribution",
                    xlabel="activation",
                )

    def _plot_dit(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return

        valid_inputs = [
            np.asarray(record.dit_model_inputs, dtype=np.float32)
            for record in self.records
            if record.dit_model_inputs
        ]
        if valid_inputs:
            num_iters = valid_inputs[0].shape[0]
            seq_len = valid_inputs[0].shape[1]
            pair_labels = []
            rows = []
            for left in range(num_iters - 1):
                right = left + 1
                pair_labels.append(f"iter_{left}_to_{right}")
                per_pos = []
                for pos in range(seq_len):
                    values = [
                        _cosine_similarity(trace[left, pos], trace[right, pos])
                        for trace in valid_inputs
                    ]
                    per_pos.append(float(np.mean(values)))
                rows.append(np.asarray(per_pos, dtype=np.float32))
            self._save_heatmap(
                output_dir / "dit_model_input_position_similarity",
                np.stack(rows, axis=0),
                "DiT Model Input Similarity Between Consecutive Iterations",
                xlabel="token_position",
                ylabel="iteration_pair",
                yticklabels=pair_labels,
            )
            flat_per_iter = {
                f"iter_{iteration}": np.concatenate(
                    [trace[iteration].reshape(-1) for trace in valid_inputs], axis=0
                )
                for iteration in range(num_iters)
            }
            self._save_multi_histogram(
                output_dir / "dit_model_input_activation_distribution",
                flat_per_iter,
                "DiT Model Input Activation Distribution by Iteration",
                xlabel="activation",
            )

        for block_name in self.selected_dit_block_names:
            block_traces = []
            for record in self.records:
                block_iters = record.dit_blocks.get(block_name)
                if not block_iters or any(item is None for item in block_iters):
                    continue
                block_traces.append(np.asarray(block_iters, dtype=np.float32))
            if not block_traces:
                continue

            num_iters = block_traces[0].shape[0]
            seq_len = block_traces[0].shape[1]
            pair_labels = []
            rows = []
            for left in range(num_iters - 1):
                right = left + 1
                pair_labels.append(f"iter_{left}_to_{right}")
                per_pos = []
                for pos in range(seq_len):
                    values = [
                        _cosine_similarity(trace[left, pos], trace[right, pos])
                        for trace in block_traces
                    ]
                    per_pos.append(float(np.mean(values)))
                rows.append(np.asarray(per_pos, dtype=np.float32))
            self._save_heatmap(
                output_dir / f"{_sanitize_name(block_name)}_position_similarity",
                np.stack(rows, axis=0),
                f"{block_name}: Input Similarity Between Consecutive Iterations",
                xlabel="token_position",
                ylabel="iteration_pair",
                yticklabels=pair_labels,
            )
            flat_per_iter = {
                f"iter_{iteration}": np.concatenate(
                    [trace[iteration].reshape(-1) for trace in block_traces], axis=0
                )
                for iteration in range(num_iters)
            }
            self._save_multi_histogram(
                output_dir / f"{_sanitize_name(block_name)}_activation_distribution",
                flat_per_iter,
                f"{block_name}: Input Activation Distribution by Iteration",
                xlabel="activation",
            )

    def _plot_requestwise_similarity(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(self.records) < 2:
            return

        image_labels = self.records[0].image_labels
        if image_labels:
            raw_image_rows = []
            for image_label in image_labels:
                vectors = [
                    record.raw_image_vectors[image_label]
                    for record in self.records
                    if image_label in record.raw_image_vectors
                ]
                raw_image_rows.append(_consecutive_cosines(vectors))
            self._save_ragged_heatmap(
                output_dir / "raw_image_consecutive_similarity",
                raw_image_rows,
                image_labels,
                "Raw Input Image Consecutive Request Cosine",
                value_label="cosine",
            )

        action_preds = [record.action_pred for record in self.records if record.action_pred is not None]
        if action_preds:
            flat_actions = np.stack(
                [np.asarray(item, dtype=np.float32).reshape(-1) for item in action_preds], axis=0
            )
            self._save_heatmap(
                output_dir / "action_pairwise_similarity",
                _pairwise_cosine_matrix(flat_actions),
                "Action Output Pairwise Request Cosine",
                xlabel="request",
                ylabel="request",
            )
            self._save_line_plot(
                output_dir / "action_consecutive_similarity",
                {"action_output": _consecutive_cosines([item.reshape(-1) for item in action_preds])},
                "Action Output Consecutive Request Cosine",
                xlabel="request_pair",
                ylabel="cosine",
            )

            horizon_rows = []
            dim_rows = []
            for left in range(len(action_preds) - 1):
                right = left + 1
                horizon_rows.append(
                    np.asarray(
                        [
                            _cosine_similarity(action_preds[left][h], action_preds[right][h])
                            for h in range(action_preds[left].shape[0])
                        ],
                        dtype=np.float32,
                    )
                )
                dim_rows.append(
                    np.asarray(
                        [
                            _cosine_similarity(
                                action_preds[left][:, d],
                                action_preds[right][:, d],
                            )
                            for d in range(action_preds[left].shape[1])
                        ],
                        dtype=np.float32,
                    )
                )
            self._save_heatmap(
                output_dir / "action_consecutive_similarity_by_horizon",
                np.stack(horizon_rows, axis=0),
                "Action Output Consecutive Similarity by Horizon",
                xlabel="action_horizon",
                ylabel="request_pair",
            )
            self._save_heatmap(
                output_dir / "action_consecutive_similarity_by_dimension",
                np.stack(dim_rows, axis=0),
                "Action Output Consecutive Similarity by Dimension",
                xlabel="action_dimension",
                ylabel="request_pair",
            )

        component_series: dict[str, list[np.ndarray]] = {}
        if self.vision_layers:
            last_layer = list(self.vision_layers.keys())[-1]
            component_series["vision_last_pooled"] = [
                np.asarray(record.vision[last_layer].pooled, dtype=np.float32).reshape(-1)
                for record in self.records
                if last_layer in record.vision
            ]
        if self.vlm_layers:
            last_layer = list(self.vlm_layers.keys())[-1]
            component_series["vlm_last_pooled"] = [
                np.asarray(record.vlm[last_layer].pooled, dtype=np.float32).reshape(-1)
                for record in self.records
                if last_layer in record.vlm
            ]
        component_series["dit_model_input_last_iter"] = [
            np.asarray(record.dit_model_inputs[-1], dtype=np.float32).reshape(-1)
            for record in self.records
            if record.dit_model_inputs
        ]
        self._save_line_plot(
            output_dir / "component_consecutive_similarity",
            {
                key: _consecutive_cosines(value)
                for key, value in component_series.items()
                if len(value) >= 2
            },
            "Consecutive Request Cosine for Representative Components",
            xlabel="request_pair",
            ylabel="cosine",
        )

    def _summarize_weights(self) -> dict[str, Any]:
        grouped_values: dict[str, list[np.ndarray]] = {}
        grouped_stats: dict[str, dict[str, float]] = {}
        for name, param in self.model.named_parameters():
            group = self._group_parameter_name(name)
            flat = param.detach().float().reshape(-1)
            if flat.numel() == 0:
                continue
            step = max(1, flat.numel() // self.config.weight_sample_size)
            sampled = flat[::step][: self.config.weight_sample_size].cpu().numpy()
            grouped_values.setdefault(group, []).append(sampled)
        for group, pieces in grouped_values.items():
            values = np.concatenate(pieces, axis=0).astype(np.float32)
            grouped_stats[group] = {
                "sample_count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "abs_mean": float(np.abs(values).mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        self._weight_samples_cache = {
            group: np.concatenate(values, axis=0).astype(np.float32)
            for group, values in grouped_values.items()
        }
        return grouped_stats

    def _group_parameter_name(self, name: str) -> str:
        if name.startswith("backbone.model.vision_model"):
            return "backbone_vision"
        if name.startswith("backbone.model.language_model"):
            return "backbone_language"
        if name.startswith("backbone.model.mlp1"):
            return "backbone_projector"
        if name.startswith("action_head.model"):
            return "action_head_dit"
        if name.startswith("action_head.state_encoder"):
            return "action_head_state_encoder"
        if name.startswith("action_head.action_encoder"):
            return "action_head_action_encoder"
        if name.startswith("action_head.action_decoder"):
            return "action_head_action_decoder"
        if name.startswith("action_head.vlln"):
            return "action_head_vlln"
        if name.startswith("action_head.position_embedding"):
            return "action_head_position_embedding"
        return "other"

    def _plot_weights(self, output_dir: Path, weight_summary: dict[str, Any]):
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "weight_summary.json", weight_summary)
        if not getattr(self, "_weight_samples_cache", None):
            return
        self._save_multi_histogram(
            output_dir / "weight_distributions",
            self._weight_samples_cache,
            "Weight Distribution by Model Part",
            xlabel="weight",
        )

    def _import_plotting(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save_heatmap(
        self,
        base_path: Path,
        matrix: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        xticklabels: list[str] | None = None,
        yticklabels: list[str] | None = None,
    ):
        if matrix.size == 0:
            return
        plt = self._import_plotting()
        fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.45), max(5, matrix.shape[0] * 0.35)))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticklabels is not None:
            ax.set_xticks(np.arange(len(xticklabels)))
            ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
        if yticklabels is not None:
            ax.set_yticks(np.arange(len(yticklabels)))
            ax.set_yticklabels(yticklabels, fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.85)
        fig.tight_layout()
        for fmt in self.plot_formats:
            fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=180)
        plt.close(fig)

    def _save_ragged_heatmap(
        self,
        base_path: Path,
        rows: list[np.ndarray],
        row_labels: list[str],
        title: str,
        value_label: str,
    ):
        if not rows:
            return
        max_len = max((row.shape[0] for row in rows), default=0)
        if max_len == 0:
            return
        matrix = np.full((len(rows), max_len), np.nan, dtype=np.float32)
        for idx, row in enumerate(rows):
            matrix[idx, : row.shape[0]] = row
        plt = self._import_plotting()
        fig, ax = plt.subplots(figsize=(max(8, max_len * 0.45), max(5, len(rows) * 0.35)))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("request_pair")
        ax.set_ylabel("series")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.85, label=value_label)
        fig.tight_layout()
        for fmt in self.plot_formats:
            fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=180)
        plt.close(fig)

    def _save_histogram(self, base_path: Path, values: np.ndarray, title: str, xlabel: str):
        if values.size == 0:
            return
        plt = self._import_plotting()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(values.astype(np.float32), bins=80, alpha=0.85, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        for fmt in self.plot_formats:
            fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=180)
        plt.close(fig)

    def _save_multi_histogram(
        self,
        base_path: Path,
        series: dict[str, np.ndarray],
        title: str,
        xlabel: str,
    ):
        if not series:
            return
        plt = self._import_plotting()
        n_rows = len(series)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(3, 2.6 * n_rows)), sharex=True)
        if n_rows == 1:
            axes = [axes]
        for ax, (label, values) in zip(axes, series.items()):
            if values.size == 0:
                continue
            ax.hist(values.astype(np.float32), bins=80, alpha=0.85)
            ax.set_ylabel(label)
        axes[0].set_title(title)
        axes[-1].set_xlabel(xlabel)
        fig.tight_layout()
        for fmt in self.plot_formats:
            fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=180)
        plt.close(fig)

    def _save_line_plot(
        self,
        base_path: Path,
        series: dict[str, np.ndarray],
        title: str,
        xlabel: str,
        ylabel: str,
    ):
        if not series:
            return
        plt = self._import_plotting()
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, values in series.items():
            if values.size == 0:
                continue
            ax.plot(np.arange(values.shape[0]), values.astype(np.float32), label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for fmt in self.plot_formats:
            fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=180)
        plt.close(fig)


class InstrumentedGr00tPolicy(PolicyWrapper):
    def __init__(self, policy: Gr00tPolicy, analyzer: InferenceAnalyzer):
        super().__init__(policy, strict=policy.strict)
        self.policy = policy
        self.analyzer = analyzer
        self.modality_configs = policy.modality_configs
        self.language_key = policy.language_key
        self.processor = policy.processor
        self.collate_fn = policy.collate_fn
        self.embodiment_tag = policy.embodiment_tag

    def __getattr__(self, name: str):
        return getattr(self.policy, name)

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self) -> dict[str, Any]:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        analysis_info = self.analyzer.reset()
        policy_info = self.policy.reset(options)
        if policy_info:
            analysis_info["policy_reset"] = policy_info
        return analysis_info

    def _sample_raw_image_vector(self, image: np.ndarray) -> np.ndarray:
        flat = image.astype(np.float32).reshape(-1) / 255.0
        idx = _evenly_spaced_indices(flat.shape[0], self.analyzer.config.raw_image_sample_size)
        return flat[idx]

    def _build_image_metadata(self, observation: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
        labels = []
        raw_vectors = {}
        video_keys = list(self.modality_configs["video"].modality_keys)
        if not video_keys:
            return labels, raw_vectors
        horizon = observation["video"][video_keys[0]].shape[1]
        for t in range(horizon):
            for video_key in video_keys:
                label = f"t{t}:{video_key}"
                labels.append(label)
                raw_vectors[label] = self._sample_raw_image_vector(observation["video"][video_key][0, t])
        return labels, raw_vectors

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unbatched_observations = self.policy._unbatch_observation(observation)
        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            vla_step_data = self.policy._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        image_labels, raw_image_vectors = self._build_image_metadata(observation)
        with torch.inference_mode():
            normalized_action = self.analyzer.run_model(
                collated_inputs,
                image_labels=image_labels,
                raw_image_vectors=raw_image_vectors,
            ).float()

        batched_states = {}
        for key in self.modality_configs["state"].modality_keys:
            batched_states[key] = np.stack([state[key] for state in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
        return casted_action, {"analysis_request_index": len(self.analyzer.records) - 1}
