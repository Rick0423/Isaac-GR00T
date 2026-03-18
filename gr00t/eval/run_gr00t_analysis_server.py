from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.inference_analysis import AnalysisConfig, InferenceAnalyzer, InstrumentedGr00tPolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class AnalysisServerConfig:
    model_path: str | None = None
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = DEFAULT_MODEL_SERVER_PORT
    strict: bool = True
    use_sim_policy_wrapper: bool = False

    analysis_output_dir: str = "/tmp/gr00t_inference_analysis"
    analysis_token_sample_size: int = 8
    analysis_dit_block_indices: str = "auto"
    analysis_weight_sample_size: int = 200000
    analysis_plot_formats: str = "png"
    analysis_auto_dump_on_exit: bool = True
    analysis_auto_dump_on_reset: bool = False
    analysis_save_raw_tensors: bool = True


def main(config: AnalysisServerConfig):
    if config.model_path is None:
        raise ValueError("model_path is required for the analysis server")

    print("Starting GR00T analysis server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Analysis output dir: {config.analysis_output_dir}")

    base_policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    analyzer = InferenceAnalyzer(
        base_policy.model,
        AnalysisConfig(
            output_dir=config.analysis_output_dir,
            token_sample_size=config.analysis_token_sample_size,
            dit_block_indices=config.analysis_dit_block_indices,
            weight_sample_size=config.analysis_weight_sample_size,
            plot_formats=config.analysis_plot_formats,
            auto_dump_on_exit=config.analysis_auto_dump_on_exit,
            auto_dump_on_reset=config.analysis_auto_dump_on_reset,
            save_raw_tensors=config.analysis_save_raw_tensors,
        ),
    )
    policy = InstrumentedGr00tPolicy(base_policy, analyzer)
    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy, strict=config.strict)

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    server.register_endpoint("dump_analysis", analyzer.dump, requires_input=False)
    server.register_endpoint("analysis_status", analyzer.status, requires_input=False)
    server.register_endpoint("reset_analysis", analyzer.reset, requires_input=False)

    dump_result = None
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down analysis server...")
    finally:
        try:
            if config.analysis_auto_dump_on_exit:
                dump_result = analyzer.dump()
                print(f"Analysis written to: {dump_result['dump_dir']}")
        finally:
            analyzer.close()

    return dump_result


if __name__ == "__main__":
    main(tyro.cli(AnalysisServerConfig))
