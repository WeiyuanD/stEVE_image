"""State/image agent construction with the original shared-head ownership."""

from math import prod

import torch
import eve_rl


def build_rnn_sac_agent(config, env_train, env_eval, checkpoint=None):
    network = config["network"]
    spaces = env_train.observation_space.spaces
    n_observations = sum(prod(space.shape) for space in spaces.values())
    n_actions = prod(env_train.action_space.shape)
    # RNNSACEnv and ImageLSTM agree on image-first flattened observations.
    image_size = prod(spaces["image"].shape) if env_train.obs == "image" else 0
    if image_size and next(iter(spaces)) != "image":
        raise ValueError("Image replay requires the image to be the first observation")
    if env_train.obs == "state":
        head = eve_rl.network.component.LSTM(
            n_layer=network["num_layers"], n_nodes=network["hidden_size"])
    else:
        head = eve_rl.network.component.ImageLSTM(
            image_shape=spaces["image"].shape,
            action_dim=prod(spaces["last_action"].shape) if "last_action" in spaces else 0,
            channels=network["cnn_channels"], feature_dim=network["feature_dim"],
            hidden_size=network["hidden_size"], num_layers=network["num_layers"],
            action_fusion=network["action_fusion"],
        )
    q1_body = eve_rl.network.component.MLP(network["mlp_hidden"])
    q2_body = eve_rl.network.component.MLP(network["mlp_hidden"])
    policy_body = eve_rl.network.component.MLP(network["mlp_hidden"])
    q1 = eve_rl.network.QNetwork(q1_body, n_observations, n_actions, head)
    q2 = eve_rl.network.QNetwork(q2_body, n_observations, n_actions, head)
    policy = eve_rl.network.GaussianPolicy(policy_body, n_observations, n_actions, head)
    learning = config["learning"]
    # Like the existing state helper: Q1 alone steps the shared observation head.
    optimizers = [eve_rl.optim.Adam(module, lr=learning["lr"])
                  for module in (q1, q2_body, policy_body)]
    schedulers = [torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=learning["lr_end_factor"],
        total_iters=learning["lr_decay_updates"]) for optimizer in optimizers]
    model = eve_rl.model.SACModel(
        lr_alpha=learning["lr"], q1=q1, q2=q2, policy=policy,
        q1_optimizer=optimizers[0], q2_optimizer=optimizers[1],
        policy_optimizer=optimizers[2], q1_scheduler=schedulers[0],
        q2_scheduler=schedulers[1], policy_scheduler=schedulers[2],
    )
    algo = eve_rl.algo.SAC(
        model, n_actions=n_actions, gamma=learning["gamma"],
        reward_scaling=learning["reward_scaling"],
        exploration_action_noise=learning["exploration_action_noise"],
        stochastic_eval=learning["stochastic_eval"],
    )
    if checkpoint is not None:
        # Load before Synchron copies the model into workers and the learner.
        algo.load_state_dicts_network(checkpoint["network_state_dicts"])
        algo.load_state_dicts_optimizer(checkpoint["optimizer_state_dicts"])
        algo.load_state_dicts_scheduler(checkpoint["scheduler_state_dicts"])
        algo.alpha = model.log_alpha.detach().exp()
        algo.update_step = checkpoint["steps"]["update"]
    execution = config["agent"]
    device = torch.device(config["device"])
    if execution["kind"] == "synchron":
        replay = eve_rl.replaybuffer.VanillaEpisodeShared(
            config["replay"]["capacity_episodes"], config["replay"]["batch_size"], device,
            image_size=image_size)
        return eve_rl.agent.Synchron(
            algo=algo, env_train=env_train, env_eval=env_eval, replay_buffer=replay,
            n_worker=execution["n_worker"],
            worker_device=torch.device(execution["worker_device"]), trainer_device=device,
            consecutive_action_steps=execution["consecutive_action_steps"],
            normalize_actions=True,
            timeout_worker_after_reaching_limit=execution["timeout_worker_after_reaching_limit"],
        )
    if execution["kind"] != "single":
        raise ValueError("agent.kind must be single or synchron")
    replay = eve_rl.replaybuffer.VanillaEpisode(
        config["replay"]["capacity_episodes"], config["replay"]["batch_size"],
        image_size=image_size)
    return eve_rl.agent.Single(
        algo, env_train, env_eval, replay, device=torch.device(config["device"]),
        consecutive_action_steps=execution["consecutive_action_steps"], normalize_actions=True)
