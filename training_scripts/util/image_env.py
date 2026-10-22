"""Prepare image history around the shared StEveEnv; retain its Gymnasium API."""

from collections import OrderedDict, deque
from datetime import timedelta
from time import perf_counter

import gymnasium as gym
import numpy as np
from eve_bench.steve_env import StEveEnv


class RNNSACEnv(StEveEnv):
    def __init__(self, obs="image", frame_stack=1, include_previous_action=True,
                 action_stack=1, mode="train", n_max_steps=200, simulation_mp=False,
                 insertion_config=None, record_video_seeds=None, image_augmentation=None):
        if obs not in ("state", "image") or mode not in ("train", "eval"):
            raise ValueError("Expected obs=state/image and mode=train/eval")
        if frame_stack < 1 or action_stack < 1:
            raise ValueError("History lengths must be positive")
        if obs == "state" and (frame_stack != 1 or action_stack != 1
                               or not include_previous_action):
            raise ValueError("State mode retains the original state observation exactly")
        self.progress_start = None
        self.record_video_seeds = list(record_video_seeds or [])
        self._video_frames = []
        self._video_observation = None
        self._record_video = False
        self.obs = obs
        self.frame_stack = frame_stack
        self.include_previous_action = include_previous_action
        self.action_stack = action_stack
        self.n_max_steps = n_max_steps
        self.simulation_mp = simulation_mp
        super().__init__(obs=obs, mode=mode, n_max_steps=n_max_steps,
                         insertion_config=insertion_config, image_augmentation=image_augmentation)
        if simulation_mp:
            self.intervention.make_mp()
        self._frames = deque(maxlen=frame_stack)
        self._actions = deque(maxlen=action_stack)

    @property
    def observation_space(self):
        if self.obs == "state":
            return gym.spaces.Dict(OrderedDict(
                (key, item.space) for key, item in self.observation.observations.items()))
        spaces = OrderedDict(image=gym.spaces.Box(
            0, 255, (self.frame_stack, 96, 96), dtype=np.uint8))
        if self.include_previous_action:
            # Existing SAC exploration adds unbounded Gaussian command noise.
            spaces["last_action"] = gym.spaces.Box(
                -np.inf, np.inf, (self.action_stack * self.action_space.low.size,),
                dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def _prepare_image(self, observation, reset=False):
        image = np.asarray(observation["image"])
        if image.shape not in ((96, 96), (96, 96, 1)):
            raise ValueError(f"Expected 96x96 grayscale pixels, got {image.shape}")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError("Pixels must be finite and in [0,255]")
        image = image.reshape(96, 96).astype(np.uint8)
        if reset:
            self._frames.clear()
        for _ in range(self.frame_stack if reset else 1):
            self._frames.append(image.copy())
        result = {"image": np.stack(self._frames)}  # oldest -> newest
        if self.include_previous_action:
            result["last_action"] = np.stack(self._actions).reshape(-1).copy()
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._record_video = seed in self.record_video_seeds
        self._video_frames = []
        self._capture_video(observation, reset=True)
        if self.obs == "state":
            return observation, info
        self._actions.clear()
        for _ in range(self.action_stack):
            self._actions.append(np.zeros(self.action_space.low.size, np.float32))
        return self._prepare_image(observation, reset=True), info

    def step(self, action):
        # eve_rl scales commands into physical units. Invert only for history.
        # Compute before step: the simulator may modify the action array in place.
        if self.obs == "image":
            low, high = self.action_space.low, self.action_space.high
            normalized = (2 * ((np.asarray(action) - low) / (high - low)) - 1)
            normalized = normalized.astype(np.float32).reshape(-1).copy()
        observation, reward, terminal, truncated, info = super().step(action)
        self._capture_video(observation)
        if self.obs == "image":
            self._actions.append(normalized)
            observation = self._prepare_image(observation)
        return observation, reward, terminal, truncated, info

    def report_episode_progress(self, phase, episode_number, steps, episode):
        if self.progress_start is None:
            return
        elapsed = timedelta(seconds=int(perf_counter() - self.progress_start))
        success = float(episode.infos[-1].get("success", 0.0))
        print(
            f" {phase:<7} E: {episode_number:<12,} I: {steps:<12,} "
            f"R: {episode.episode_reward:<12.1f} S: {success:<12.1f} T: {elapsed}",
            flush=True,
        )

    def _capture_video(self, observation, reset=False):
        if not self._record_video:
            return
        if self.obs == "image":
            frame = observation["image"]
        else:
            if self._video_observation is None:
                from eve.observation import Image
                from eve.observation.imagewrapper import ShowVessels
                self._video_observation = ShowVessels(self.intervention, Image(self.intervention))
            if reset:
                self._video_observation.reset()
            else:
                self._video_observation.step()
            frame = self._video_observation.obs
        self._video_frames.append(np.asarray(frame).reshape(96, 96).astype(np.uint8).copy())

    def pop_episode_video(self):
        frames, self._video_frames = self._video_frames, []
        return frames, self.intervention.fluoroscopy.image_frequency

    def get_config_dict(self, eve_classes_only=False):
        # Preserve adapter construction rather than serializing only eve.Env.
        return dict(_class="training_scripts.util.image_env.RNNSACEnv", _id=id(self),
                    obs=self.obs, frame_stack=self.frame_stack,
                    include_previous_action=self.include_previous_action,
                    action_stack=self.action_stack, mode=self.mode,
                    n_max_steps=self.n_max_steps, simulation_mp=self.simulation_mp,
                    insertion_config=self.insertion_config,
                    record_video_seeds=self.record_video_seeds,
                    image_augmentation=self.image_augmentation)
