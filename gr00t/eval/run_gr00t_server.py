from dataclasses import dataclass
import json
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.inference_analysis import AnalysisConfig, InferenceAnalyzer, InstrumentedGr00tPolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    enable_inference_analysis: bool = False
    """Enable server-side inference analysis and register analysis endpoints"""

    analysis_output_dir: str = "/tmp/gr00t_inference_analysis"
    """Directory where analysis dumps are written"""

    analysis_token_sample_size: int = 12
    """Number of spatial tokens sampled per image for similarity heatmaps"""

    analysis_raw_image_sample_size: int = 4096
    """Number of raw input pixel values sampled per image for request-wise similarity"""

    analysis_detail_layer_indices: str = "auto"
    """Representative layer indices to dump detailed per-image spatial heatmaps for"""

    analysis_detail_request_indices: str = "auto"
    """Representative request indices to dump detailed per-image spatial heatmaps for"""

    analysis_dit_block_indices: str = "auto"
    """Which DiT transformer blocks to trace in detail"""

    analysis_weight_sample_size: int = 200000
    """Maximum sampled weights per model part when plotting distributions"""

    analysis_plot_formats: str = "png"
    """Comma-separated list of output figure formats"""

    analysis_auto_dump_on_exit: bool = True
    """Dump analysis automatically when the server exits"""

    analysis_auto_dump_on_reset: bool = False
    """Dump and clear analysis buffers when the client calls reset()"""

    analysis_save_raw_tensors: bool = True
    """Persist raw trace tensors in addition to summary plots"""


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Inference analysis: {config.enable_inference_analysis}")
    if config.enable_inference_analysis:
        print(f"  Analysis output dir: {config.analysis_output_dir}")

    # check if the model path exists
    if config.model_path and config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    analyzer = None

    # Create and start the server
    if config.model_path is not None:
        base_policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        if config.enable_inference_analysis:
            analyzer = InferenceAnalyzer(
                base_policy.model,
                AnalysisConfig(
                    output_dir=config.analysis_output_dir,
                    token_sample_size=config.analysis_token_sample_size,
                    raw_image_sample_size=config.analysis_raw_image_sample_size,
                    detail_layer_indices=config.analysis_detail_layer_indices,
                    detail_request_indices=config.analysis_detail_request_indices,
                    dit_block_indices=config.analysis_dit_block_indices,
                    weight_sample_size=config.analysis_weight_sample_size,
                    plot_formats=config.analysis_plot_formats,
                    auto_dump_on_exit=config.analysis_auto_dump_on_exit,
                    auto_dump_on_reset=config.analysis_auto_dump_on_reset,
                    save_raw_tensors=config.analysis_save_raw_tensors,
                ),
            )
            policy = InstrumentedGr00tPolicy(base_policy, analyzer)
        else:
            policy = base_policy
    elif config.dataset_path is not None:
        if config.enable_inference_analysis:
            raise ValueError("Inference analysis mode currently supports model_path only, not dataset replay")
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )
    if analyzer is not None:
        server.register_endpoint("dump_analysis", analyzer.dump, requires_input=False)
        server.register_endpoint("analysis_status", analyzer.status, requires_input=False)
        server.register_endpoint("reset_analysis", analyzer.reset, requires_input=False)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if analyzer is not None:
            try:
                if config.analysis_auto_dump_on_exit:
                    dump_result = analyzer.dump()
                    print(f"Analysis written to: {dump_result['dump_dir']}")
            finally:
                analyzer.close()


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
