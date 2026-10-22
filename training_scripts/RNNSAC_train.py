"""Run with python -m training_scripts.RNNSAC_train --config ... --output ..."""

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import random
import sys
from time import perf_counter

# The repository folder contains the actual eve_rl package one level deeper.
# Prefer this checkout (including its reporting hooks and ImageLSTM) over an
# outer namespace package or another editable installation. Spawned workers
# inherit sys.path and execute this setup before importing the agent helpers.
_eve_rl_source = Path(__file__).resolve().parents[1] / "eve_rl"
if not (_eve_rl_source / "eve_rl" / "__init__.py").is_file():
    raise ImportError(f"Missing local eve_rl package in {_eve_rl_source}")
sys.path.insert(0, str(_eve_rl_source))

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from importlib.util import find_spec

from training_scripts.util.image_agent import build_rnn_sac_agent
from training_scripts.util.image_env import RNNSACEnv
from training_scripts.util.rnn_sac_logging import LoggedRunner, TrainingLogger
from training_scripts.util.rnn_sac_resume import (
    read_resume_checkpoint, restore_counters, refill_replay,
)


def load_config(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as stream:
        override = yaml.safe_load(stream)
    if not isinstance(override, dict):
        raise ValueError("Configuration must be a YAML mapping")
    base = override.pop("extends", None)
    config = load_config(path.parent / base, (*ancestors, path)) if base else {}

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)

    merge(config, override)
    return config


def validate_config(config):
    # Older saved configs used a single limit for both environments.
    config["env"].setdefault("eval_n_max_steps", config["env"]["n_max_steps"])
    config["env"].setdefault("image_augmentation", None)
    from eve.observation.imagewrapper.imageaugmentation import ImageAugmentation
    ImageAugmentation(config["env"]["image_augmentation"])  # Validate before creating simulators.
    sections = {
        "logging": {"enable_wandb", "project", "entity", "name", "group", "mode",
                    "save_csv", "save_video", "video_eval_seeds", "save_agent"},
        "env": {"obs", "frame_stack", "include_previous_action", "action_stack", "mode", "eval_mode", "n_max_steps", "eval_n_max_steps", "simulation_mp", "insertion_config", "image_augmentation"},
        "agent": {"kind", "n_worker", "worker_device", "consecutive_action_steps", "timeout_worker_after_reaching_limit"},
        "runner": {"heatup_action_low", "heatup_action_high", "info_results"},
        "network": {"cnn_channels", "feature_dim", "action_fusion", "hidden_size", "num_layers", "mlp_hidden"},
        "learning": {"lr", "lr_end_factor", "lr_decay_updates", "gamma", "reward_scaling", "exploration_action_noise", "stochastic_eval"},
        "replay": {"capacity_episodes", "batch_size"},
        "training": {"heatup_steps", "training_steps", "explore_steps_between_eval",
                     "explore_episodes_between_updates", "update_steps_per_explore_step", "eval_seeds"},
    }
    if set(config) != set(sections) | {"seed", "device"}:
        raise ValueError("Unexpected or missing top-level configuration fields")
    for section, keys in sections.items():
        if set(config[section]) != keys:
            raise ValueError(f"Expected {section} fields: {sorted(keys)}")
    positive_integers = {
        "env": ("frame_stack", "action_stack", "n_max_steps", "eval_n_max_steps"),
        "agent": ("n_worker", "consecutive_action_steps", "timeout_worker_after_reaching_limit"),
        "network": ("feature_dim", "hidden_size", "num_layers"),
        "learning": ("lr_decay_updates",),
        "replay": ("capacity_episodes", "batch_size"),
        "training": ("heatup_steps", "training_steps", "explore_steps_between_eval",
                     "explore_episodes_between_updates"),
    }
    for section, keys in positive_integers.items():
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    env, network, learning = config["env"], config["network"], config["learning"]
    if env["obs"] not in ("state", "image") or env["mode"] not in ("train", "eval"):
        raise ValueError("Expected env.obs=state/image and env.mode=train/eval")
    if env["eval_mode"] not in ("train", "eval") or type(env["simulation_mp"]) is not bool:
        raise ValueError("Invalid evaluation mode or simulation multiprocessing switch")
    if config["agent"]["kind"] not in ("single", "synchron"):
        raise ValueError("agent.kind must be single or synchron")
    if type(env["include_previous_action"]) is not bool or type(learning["stochastic_eval"]) is not bool:
        raise ValueError("Observation and evaluation switches must be booleans")
    if env["obs"] == "state" and (env["frame_stack"] != 1 or env["action_stack"] != 1
                                  or not env["include_previous_action"]):
        raise ValueError("State mode retains its existing observation without extra history")
    if network["action_fusion"] not in ("concat", "sum"):
        raise ValueError("network.action_fusion must be concat or sum")
    for key, minimum in (("mlp_hidden", 2), ("cnn_channels", 1)):
        widths = network[key]
        if not isinstance(widths, list) or len(widths) < minimum or any(
                type(width) is not int or width < 1 for width in widths):
            raise ValueError(f"network.{key} needs at least {minimum} positive integer widths")
    if env["obs"] == "image" and 96 // (2 ** len(network["cnn_channels"])) < 1:
        raise ValueError("Too many CNN pooling blocks for 96x96 images")
    if not 0 <= learning["gamma"] <= 1 or learning["lr"] <= 0:
        raise ValueError("Expected gamma in [0,1] and positive learning rate")
    if not 0 < learning["lr_end_factor"] <= 1 or learning["exploration_action_noise"] < 0:
        raise ValueError("Invalid learning-rate decay or exploration noise")
    if config["replay"]["capacity_episodes"] <= config["replay"]["batch_size"]:
        raise ValueError("Replay capacity must exceed batch size")
    if config["training"]["update_steps_per_explore_step"] <= 0:
        raise ValueError("Update ratio must be positive")
    seeds = config["training"]["eval_seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Provide a nonempty list of nonnegative evaluation seeds")
    if config["seed"] is not None and (type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32):
        raise ValueError("seed must be null or an integer in [0,2**32)")
    log = config["logging"]
    for key in ("enable_wandb", "save_csv", "save_video", "save_agent"):
        if type(log[key]) is not bool:
            raise ValueError(f"logging.{key} must be a boolean")
    if log["mode"] not in ("online", "offline"):
        raise ValueError("logging.mode must be online or offline")
    video_seeds = log["video_eval_seeds"]
    if video_seeds is not None and (not isinstance(video_seeds, list) or any(
            type(seed) is not int or seed not in seeds for seed in video_seeds)):
        raise ValueError("logging.video_eval_seeds must be null or a subset of evaluation seeds")
    for enabled, modules in ((log["enable_wandb"], ("wandb",)),
                             (log["save_video"], ("imageio", "imageio_ffmpeg"))):
        if enabled:
            for module in modules:
                if find_spec(module) is None:
                    raise ValueError(f"Install {module} to enable the requested logging feature")
    runner = config["runner"]
    low, high = runner["heatup_action_low"], runner["heatup_action_high"]
    if (low is None) != (high is None):
        raise ValueError("Specify both heatup bounds or leave both null")
    if low is not None:
        low, high = np.asarray(low), np.asarray(high)
        if low.ndim != 1 or low.shape != high.shape or not (
                np.isfinite(low).all() and np.isfinite(high).all() and (low < high).all()):
            raise ValueError("Heatup bounds must be finite matching vectors with low < high")
    if runner["info_results"] is not None and not isinstance(runner["info_results"], list):
        raise ValueError("runner.info_results must be a list or null")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--resume", type=Path, help="Trusted local .everl checkpoint")
    parser.add_argument("--resume-refill-episodes", type=int, default=None,
                        help="Random episodes to refill replay; default: batch_size + 1")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    checkpoint = best_eval = None
    if args.resume is not None:
        checkpoint, best_eval = read_resume_checkpoint(args.resume, config)
        refill_episodes = args.resume_refill_episodes
        if refill_episodes is None:
            refill_episodes = config["replay"]["batch_size"] + 1
        if refill_episodes <= config["replay"]["batch_size"]:
            parser.error("--resume-refill-episodes must exceed replay.batch_size")
    elif args.resume_refill_episodes is not None:
        parser.error("--resume-refill-episodes requires --resume")
    # Set before creating shared replay, workers, or simulator subprocesses.
    mp.set_start_method("spawn", force=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    with (args.output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(), logging.FileHandler(args.output / "training.log")])
    if config["seed"] is not None:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
    env_train = env_eval = agent = metrics_logger = None
    completed = False
    try:
        env_config = dict(config["env"])
        eval_mode = env_config.pop("eval_mode")
        eval_n_max_steps = env_config.pop("eval_n_max_steps")
        env_train = RNNSACEnv(**env_config)
        video_seeds = config["logging"]["video_eval_seeds"]
        if video_seeds is None:
            video_seeds = config["training"]["eval_seeds"][:2]
        env_eval = RNNSACEnv(**dict(env_config, mode=eval_mode, n_max_steps=eval_n_max_steps),
                            record_video_seeds=video_seeds if config["logging"]["save_video"] else [])
        for name, env in (("env_train", env_train), ("env_eval", env_eval)):
            with (args.output / f"{name}.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump(env.get_config_dict(), stream, sort_keys=False)
        runner_config = config["runner"]
        heatup_low = runner_config["heatup_action_low"]
        heatup_high = runner_config["heatup_action_high"]
        if heatup_low is None:
            heatup_low = env_train.action_space.low.reshape(-1).tolist()
            heatup_high = env_train.action_space.high.reshape(-1).tolist()
        if len(heatup_low) != env_train.action_space.low.size or (
                np.any(np.asarray(heatup_low) < env_train.action_space.low.reshape(-1))
                or np.any(np.asarray(heatup_high) > env_train.action_space.high.reshape(-1))):
            raise ValueError("Heatup bounds must fit the environment action space")
        info_results = runner_config["info_results"]
        if info_results is None:
            info_results = list(env_eval.info.info.keys())
        # Copied into worker environments: one common clock for live progress.
        env_train.progress_start = perf_counter()
        agent = build_rnn_sac_agent(config, env_train, env_eval, checkpoint=checkpoint)
        if checkpoint is not None:
            restore_counters(agent, checkpoint)
        # Start W&B only in the parent, after agent workers have been spawned.
        metrics_logger = TrainingLogger(config, args.output)
        runner = LoggedRunner(
            metrics_logger=metrics_logger,
            agent=agent,
            heatup_action_low=heatup_low,
            heatup_action_high=heatup_high,
            agent_parameter_for_result_file={
                "observation": config["env"]["obs"],
                "frame_stack": config["env"]["frame_stack"],
                "action_fusion": config["network"]["action_fusion"],
            },
            checkpoint_folder=str(args.output / "checkpoints"),
            results_file=str(args.output / "results.csv"),
            quality_info="success", info_results=info_results,
        )
        training = dict(config["training"])
        if checkpoint is not None:
            runner.best_eval = best_eval
            with (args.output / "resume.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump({"checkpoint": str(args.resume.resolve()),
                                "restored_steps": checkpoint["steps"],
                                "refill_episodes": refill_episodes,
                                "best_eval": best_eval}, stream)
            refill_replay(runner, refill_episodes)
            training["heatup_steps"] = 0  # Refill replaces the original full heatup.
        runner.training_run(**training)
        completed = True
    finally:
        try:
            if agent is not None:
                agent.close()
            else:
                if env_train is not None:
                    env_train.close()
                if env_eval is not None:
                    env_eval.close()
        finally:
            if metrics_logger is not None:
                metrics_logger.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
