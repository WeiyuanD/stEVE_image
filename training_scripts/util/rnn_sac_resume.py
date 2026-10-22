"""Restore trusted local training checkpoints; replay is intentionally rebuilt."""

import logging
from pathlib import Path
from time import monotonic, sleep

import torch
import yaml


def read_resume_checkpoint(path, config):
    path = Path(path).resolve()
    # Checkpoints contain Python config objects as well as tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("network_state_dicts", "optimizer_state_dicts", "scheduler_state_dicts",
                "steps", "episodes"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing {key}")
    previous_config = path.parent.parent / "config.yaml"
    if previous_config.is_file():
        with previous_config.open(encoding="utf-8") as stream:
            previous = yaml.safe_load(stream)
        previous["env"].setdefault("image_augmentation", None)
        previous["env"].setdefault("eval_n_max_steps", previous["env"]["n_max_steps"])
        # Execution/logging/schedule can change; observation and learning must match.
        for section in ("env", "network", "learning"):
            if previous[section] != config[section]:
                raise ValueError(f"Resume requires matching {section}; use {previous_config}")
    else:
        logging.getLogger(__name__).warning(
            "No original config.yaml found; ensure observation and learning settings match")
    if checkpoint["steps"]["exploration"] >= config["training"]["training_steps"]:
        raise ValueError("training_steps must exceed the checkpoint's exploration step count")
    info = checkpoint.get("additional_info") or {}
    best = info.get("best_eval")
    if best is None:
        best = {"quality": info.get("quality", float("-inf")),
                "steps": checkpoint["steps"]["exploration"]}
        # Older checkpoints omitted the historical best score from metadata.
        sibling = path.parent / "best_checkpoint.everl"
        if sibling.is_file() and sibling != path:
            old_best = torch.load(sibling, map_location="cpu", weights_only=False)
            best_step = old_best["steps"]["exploration"]
            best_quality = (old_best.get("additional_info") or {}).get("quality", float("-inf"))
            if best_step <= checkpoint["steps"]["exploration"] and best_quality > best["quality"]:
                best = {"quality": best_quality, "steps": best_step}
    return checkpoint, dict(best)


def restore_counters(agent, checkpoint):
    # Assign fields rather than replacing shared counters referenced by workers.
    for key in ("heatup", "exploration", "evaluation", "update"):
        setattr(agent.step_counter, key, checkpoint["steps"][key])
    for key in ("heatup", "exploration", "evaluation"):
        setattr(agent.episode_counter, key, checkpoint["episodes"][key])


def refill_replay(runner, episodes):
    runner.logger.info(
        "Restored exploration step %s, update %s. Refilling empty replay with %s "
        "random-action episodes before learning; these count as additional heatup.",
        runner.step_counter.exploration, runner.step_counter.update, episodes,
    )
    runner.agent.heatup(episodes=episodes, custom_action_low=runner.heatup_action_low,
                        custom_action_high=runner.heatup_action_high)
    # Shared replay consumes pushed episodes asynchronously. Let it drain the
    # queue before checking readiness (no learning or extra resets while waiting).
    deadline = monotonic() + 30
    while len(runner.agent.replay_buffer) <= runner.agent.replay_buffer.batch_size:
        if monotonic() >= deadline:
            raise RuntimeError("Replay refill returned too few episodes to resume learning")
        sleep(0.1)
