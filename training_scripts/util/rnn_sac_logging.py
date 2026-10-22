"""Parent-process reporting for RNN-SAC; workers only return episode data."""

import csv
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from eve_rl import Runner


def mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else 0.0


class TrainingLogger:
    def __init__(self, config, output):
        self.output = Path(output)
        self.settings = config["logging"]
        self.start = perf_counter()
        self.run = None
        self.wandb = None
        if self.settings["enable_wandb"]:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=self.settings["project"], entity=self.settings["entity"],
                name=self.settings["name"] or self.output.name,
                group=self.settings["group"] or f'rnn-sac-{config["env"]["obs"]}-{config["env"]["frame_stack"]}',
                config=config, dir=str(self.output.resolve()), mode=self.settings["mode"],
            )
            for prefix, axis in (("train", "train/step"), ("loss", "train/step"),
                                 ("eval", "eval/step"), ("videos", "eval/step")):
                self.run.define_metric(axis)
                self.run.define_metric(f"{prefix}/*", step_metric=axis)

    def csv(self, name, row):
        if not self.settings["save_csv"]:
            return
        path = self.output / name
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def log(self, name, row):
        self.csv(name, row)
        if self.run is not None:
            self.run.log(row)

    def video(self, episode, step):
        frames = getattr(episode, "video_frames", None)
        if not frames or not self.settings["save_video"]:
            return
        try:
            # Call the encoder directly: older imageio-ffmpeg releases do not
            # accept the audio kwargs passed unconditionally by newer imageio.
            import imageio_ffmpeg
            folder = self.output / "videos"
            folder.mkdir(exist_ok=True)
            path = folder / f"eval_{step}_seed_{episode.seed}.mp4"
            temporary = path.with_suffix(".partial.mp4")
            height, width = np.asarray(frames[0]).shape
            writer = imageio_ffmpeg.write_frames(
                str(temporary), (width, height), fps=float(episode.video_fps),
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            )
            try:
                writer.send(None)
                for frame in frames:
                    rgb = np.repeat(np.asarray(frame, dtype=np.uint8)[..., None], 3, axis=-1)
                    writer.send(np.ascontiguousarray(rgb).tobytes())
            finally:
                writer.close()
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Video encoder produced no output")
            temporary.replace(path)
            if self.run is not None:
                self.run.log({"eval/step": step, f"videos/seed_{episode.seed}":
                              self.wandb.Video(str(path), format="mp4")})
        except Exception:
            # Optional media must never terminate a long training run or
            # prevent the following best-checkpoint artifact upload.
            logging.getLogger(__name__).exception(
                "Video export/upload failed at step %s, seed %s; continuing training",
                step, episode.seed,
            )

    def checkpoint(self, path, step):
        if self.run is not None and self.settings["save_agent"]:
            artifact = self.wandb.Artifact(f"rnn-sac-{self.run.id}", type="model",
                                           metadata={"exploration_step": step})
            artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=["best"])

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


class LoggedRunner(Runner):
    def __init__(self, *args, metrics_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_logger = metrics_logger

    def on_training_batch(self, episodes, losses):
        elapsed = perf_counter() - self.metrics_logger.start
        row = {
            "train/step": self.step_counter.exploration,
            "train/episode": self.episode_counter.exploration,
            "train/update": self.step_counter.update,
            "train/heatup_steps": self.step_counter.heatup,
            "train/episode_reward": mean([ep.episode_reward for ep in episodes]),
            "train/episode_length": mean([len(ep.actions) for ep in episodes]),
            "train/success_rate": mean([ep.infos[-1]["success"] for ep in episodes]),
            "train/elapsed_time": elapsed,
            "train/steps_per_second": self.step_counter.exploration / max(elapsed, 1e-9),
        }
        model = self.agent.algo.model
        row["train/alpha"] = float(model.log_alpha.detach().exp().cpu().item())
        for name in ("q1", "q2", "policy"):
            optimizer = getattr(model, f"{name}_optimizer")
            row[f"train/lr_{name}"] = float(optimizer.param_groups[0]["lr"])
        # SAC returns one [Q1, Q2, policy] loss vector per update.
        loss_array = np.asarray(losses if losses is not None else [], dtype=float)
        loss_row = {"train/step": self.step_counter.exploration,
                    "train/update": self.step_counter.update}
        if loss_array.size:
            for name, value in zip(("q1", "q2", "policy"), loss_array.reshape(-1, 3).mean(axis=0)):
                loss_row[f"loss/{name}"] = float(value)
            self.metrics_logger.log("loss.csv", loss_row)
        self.metrics_logger.log("train.csv", row)

    def on_evaluation(self, episodes, save_best):
        step = self.step_counter.exploration
        rows = []
        for ep in episodes:
            info = ep.infos[-1]
            row = {"eval_step": step, "seed": ep.seed, "success": bool(info["success"]),
                   "episode_reward": ep.episode_reward, "episode_length": len(ep.actions),
                   "path_ratio": float(info["path_ratio"]),
                   "navigation_time": ep.wall_time_seconds}
            rows.append(row)
            self.metrics_logger.csv("eval_episodes.csv", row)
        row = {
            "eval/step": step,
            "eval/episode_reward": mean([r["episode_reward"] for r in rows]),
            "eval/episode_length": mean([r["episode_length"] for r in rows]),
            "eval/success_rate": mean([r["success"] for r in rows]),
            # Match TD-MPC2: failed episodes for path ratio; successes for wall time.
            "eval/path_ratio": mean([r["path_ratio"] for r in rows if not r["success"]]),
            "eval/navigation_time": mean([r["navigation_time"] for r in rows if r["success"]]),
            "eval/best_success_rate": self.best_eval["quality"],
            "eval/elapsed_time": perf_counter() - self.metrics_logger.start,
        }
        self.metrics_logger.log("eval.csv", row)
        for episode in episodes:
            self.metrics_logger.video(episode, step)
        if save_best:
            self.metrics_logger.checkpoint(Path(self.checkpoint_folder) / "best_checkpoint.everl", step)
