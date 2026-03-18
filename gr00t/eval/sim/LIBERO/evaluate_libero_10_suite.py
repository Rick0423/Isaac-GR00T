import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all tasks in the LIBERO-10 suite against a running GR00T server."
    )
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--n_envs", type=int, default=5)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--task_suite", type=str, default="libero_10")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--single_task_env_name", type=str, default="")
    return parser.parse_args()


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def get_suite_env_names(task_suite_name: str) -> list[str]:
    from libero.libero import benchmark

    task_suite_name = task_suite_name.lower()
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite '{task_suite_name}'. Available: {sorted(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[task_suite_name]()
    return [f"libero_sim/{task_name}" for task_name in task_suite.get_task_names()]


def summarize_result(env_name: str, episode_successes: list[Any]) -> dict[str, Any]:
    successes = [bool(success) for success in episode_successes]
    n_episodes = len(successes)
    n_successes = int(sum(successes))
    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "env_name": env_name,
        "n_episodes": n_episodes,
        "n_successes": n_successes,
        "success_rate": success_rate,
        "episode_successes": successes,
    }


def run_single_task(args: argparse.Namespace) -> dict[str, Any]:
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    results = run_gr00t_sim_policy(
        env_name=args.single_task_env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path="",
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
    )
    summary = summarize_result(results[0], results[1])
    print(f"JSON_SUMMARY: {json.dumps(summary, ensure_ascii=True, sort_keys=True)}")
    return summary


def stream_subprocess(command: list[str], cwd: Path, prefix: str) -> tuple[int, dict[str, Any] | None]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    summary = None
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith("JSON_SUMMARY: "):
            payload = line[len("JSON_SUMMARY: ") :].strip()
            summary = json.loads(payload)
        print(f"[{prefix}] {line}", end="")
    return_code = process.wait()
    return return_code, summary


def run_single_task_subprocess(args: argparse.Namespace, env_name: str) -> dict[str, Any]:
    repo_root = get_repo_root()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single_task_env_name",
        env_name,
        "--policy_client_host",
        args.policy_client_host,
        "--policy_client_port",
        str(args.policy_client_port),
        "--n_episodes",
        str(args.n_episodes),
        "--max_episode_steps",
        str(args.max_episode_steps),
        "--n_envs",
        str(args.n_envs),
        "--n_action_steps",
        str(args.n_action_steps),
        "--task_suite",
        args.task_suite,
    ]
    return_code, summary = stream_subprocess(command, cwd=repo_root, prefix=env_name.rsplit("/", 1)[-1])
    if return_code != 0:
        raise RuntimeError(f"Subprocess failed for {env_name} with exit code {return_code}")
    if summary is None:
        raise RuntimeError(f"Missing JSON summary for {env_name}")
    return summary


def print_batch_summary(task_suite: str, task_summaries: list[dict[str, Any]], failures: list[dict[str, str]]):
    total_episodes = sum(item["n_episodes"] for item in task_summaries)
    total_successes = sum(item["n_successes"] for item in task_summaries)
    overall_success_rate = float(total_successes / total_episodes) if total_episodes else 0.0

    print("\n=== Per-task success rate ===")
    for item in task_summaries:
        print(
            f"{item['env_name']}: {item['n_successes']}/{item['n_episodes']} "
            f"({item['success_rate']:.4f})"
        )

    if failures:
        print("\n=== Failed tasks ===")
        for failure in failures:
            print(f"{failure['env_name']}: {failure['error']}")

    print("\n=== Overall summary ===")
    print(f"task_suite: {task_suite}")
    print(f"completed_tasks: {len(task_summaries)}/{len(task_summaries) + len(failures)}")
    print(f"total_successes: {total_successes}")
    print(f"total_episodes: {total_episodes}")
    print(f"overall_success_rate: {overall_success_rate:.4f}")


def maybe_write_json(
    output_json: str,
    task_suite: str,
    args: argparse.Namespace,
    task_summaries: list[dict[str, Any]],
    failures: list[dict[str, str]],
):
    if not output_json:
        return

    total_episodes = sum(item["n_episodes"] for item in task_summaries)
    total_successes = sum(item["n_successes"] for item in task_summaries)
    payload = {
        "task_suite": task_suite,
        "policy_client_host": args.policy_client_host,
        "policy_client_port": args.policy_client_port,
        "n_episodes": args.n_episodes,
        "max_episode_steps": args.max_episode_steps,
        "n_envs": args.n_envs,
        "n_action_steps": args.n_action_steps,
        "completed_tasks": len(task_summaries),
        "failed_tasks": failures,
        "total_tasks": len(task_summaries) + len(failures),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "tasks": task_summaries,
    }
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Saved JSON summary to: {output_path}")


def run_batch(args: argparse.Namespace) -> int:
    env_names = get_suite_env_names(args.task_suite)
    task_summaries = []
    failures = []

    print(f"Evaluating task suite '{args.task_suite}' with {len(env_names)} tasks")
    print(
        f"Using running GR00T server at {args.policy_client_host}:{args.policy_client_port}, "
        f"episodes={args.n_episodes}, n_envs={args.n_envs}, max_episode_steps={args.max_episode_steps}, "
        f"n_action_steps={args.n_action_steps}"
    )

    for index, env_name in enumerate(env_names, start=1):
        print(f"\n===== [{index}/{len(env_names)}] {env_name} =====")
        try:
            summary = run_single_task_subprocess(args, env_name)
            task_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            failures.append({"env_name": env_name, "error": str(exc)})
            print(f"[error] {env_name}: {exc}")
            if args.fail_fast:
                break

    print_batch_summary(args.task_suite, task_summaries, failures)
    maybe_write_json(args.output_json, args.task_suite, args, task_summaries, failures)
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.single_task_env_name:
        run_single_task(args)
        return 0
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
