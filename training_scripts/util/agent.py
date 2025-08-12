from torch import optim
import eve_rl.eve_rl as eve_rl
import eve
import numpy as np


class BenchAgentSingle(eve_rl.agent.Single):
    def __init__(
        self,
        device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        stochastic_eval: bool = False,
        ff_only: bool = False,
    ):

        obs_dict = env_train.observation_space.sample()
        obs_list = [obs.flatten() for obs in obs_dict.values()]
        obs_np = np.concatenate(obs_list)

        n_observations = obs_np.shape[0]
        n_actions = env_train.action_space.sample().flatten().shape[0]
        if embedder_layers and embedder_nodes and not ff_only:
            q1_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
        elif embedder_layers and embedder_nodes and ff_only:
            hidden_layers = [embedder_nodes] * embedder_layers
            q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
        else:
            q1_embedder = eve_rl.network.component.ComponentDummy()

        q1_base = eve_rl.network.component.MLP(hidden_layers)
        q2_base = eve_rl.network.component.MLP(hidden_layers)
        policy_base = eve_rl.network.component.MLP(hidden_layers)

        q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
        q1_optim = eve_rl.optim.Adam(
            q1,
            lr=lr,
        )
        q1_scheduler = optim.lr_scheduler.LinearLR(
            q1_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
        q2_optim = eve_rl.optim.Adam(
            q2_base,
            lr=lr,
        )
        q2_scheduler = optim.lr_scheduler.LinearLR(
            q2_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        policy = eve_rl.network.GaussianPolicy(
            policy_base, n_observations, n_actions, q1_embedder
        )
        policy_optim = eve_rl.optim.Adam(
            policy_base,
            lr=lr,
        )
        policy_scheduler = optim.lr_scheduler.LinearLR(
            policy_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        sac_model = eve_rl.model.SACModel(
            lr_alpha=lr,
            q1=q1,
            q2=q2,
            policy=policy,
            q1_optimizer=q1_optim,
            q2_optimizer=q2_optim,
            policy_optimizer=policy_optim,
            q1_scheduler=q1_scheduler,
            q2_scheduler=q2_scheduler,
            policy_scheduler=policy_scheduler,
        )

        algo = eve_rl.algo.SAC(
            sac_model,
            n_actions=n_actions,
            gamma=gamma,
            reward_scaling=reward_scaling,
            stochastic_eval=stochastic_eval,
        )

        replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
            replay_buffer_size, batch_size, device
        )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device,
            normalize_actions=True,
        )


class BenchAgentSynchron(eve_rl.agent.Synchron):
    def __init__(
        self,
        trainer_device,
        worker_device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        n_worker,
        stochastic_eval: bool = False,
        ff_only: bool = False,
    ):

        obs_dict = env_train.observation_space.sample()
        obs_list = [obs.flatten() for obs in obs_dict.values()]
        obs_np = np.concatenate(obs_list)

        n_observations = obs_np.shape[0]
        n_actions = env_train.action_space.sample().flatten().shape[0]
        if embedder_layers and embedder_nodes and not ff_only:
            q1_embedder = eve_rl.network.component.LSTM(
                n_layer=embedder_layers, n_nodes=embedder_nodes
            )
        elif embedder_layers and embedder_nodes and ff_only:
            hidden_layers = [embedder_nodes] * embedder_layers
            q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
        else:
            q1_embedder = eve_rl.network.component.ComponentDummy()

        q1_base = eve_rl.network.component.MLP(hidden_layers)
        q2_base = eve_rl.network.component.MLP(hidden_layers)
        policy_base = eve_rl.network.component.MLP(hidden_layers)

        q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
        q1_optim = eve_rl.optim.Adam(
            q1,
            lr=lr,
        )
        q1_scheduler = optim.lr_scheduler.LinearLR(
            q1_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
        q2_optim = eve_rl.optim.Adam(
            q2_base,
            lr=lr,
        )
        q2_scheduler = optim.lr_scheduler.LinearLR(
            q2_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        policy = eve_rl.network.GaussianPolicy(
            policy_base, n_observations, n_actions, q1_embedder
        )
        policy_optim = eve_rl.optim.Adam(
            policy_base,
            lr=lr,
        )
        policy_scheduler = optim.lr_scheduler.LinearLR(
            policy_optim,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_linear_end_steps,
        )

        sac_model = eve_rl.model.SACModel(
            lr_alpha=lr,
            q1=q1,
            q2=q2,
            policy=policy,
            q1_optimizer=q1_optim,
            q2_optimizer=q2_optim,
            policy_optimizer=policy_optim,
            q1_scheduler=q1_scheduler,
            q2_scheduler=q2_scheduler,
            policy_scheduler=policy_scheduler,
        )

        algo = eve_rl.algo.SAC(
            sac_model,
            n_actions=n_actions,
            gamma=gamma,
            reward_scaling=reward_scaling,
            stochastic_eval=stochastic_eval,
        )

        replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
            replay_buffer_size, batch_size, trainer_device
        )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=trainer_device,
            worker_device=worker_device,
            n_worker=n_worker,
            normalize_actions=True,
            timeout_worker_after_reaching_limit=180,
        )


def create_bench_agent(
    device_trainer,
    device_worker,
    lr,
    lr_end_factor,
    lr_linear_end_steps,
    hidden_layers,
    embedder_nodes,
    embedder_layers,
    gamma,
    batch_size,
    reward_scaling,
    replay_buffer_size,
    train_env: eve.Env,
    eval_env: eve.Env,
    consecutive_action_steps,
    n_worker,
    stochastic_eval: bool = False,
    single: bool = False,
    ff_only: bool = False,
):
    obs_dict = train_env.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    obs_np = np.concatenate(obs_list)

    n_observations = obs_np.shape[0]
    n_actions = train_env.action_space.sample().flatten().shape[0]
    if embedder_layers and embedder_nodes and not ff_only:
        q1_embedder = eve_rl.network.component.LSTM(
            n_layer=embedder_layers, n_nodes=embedder_nodes
        )
    elif embedder_layers and embedder_nodes and ff_only:
        hidden_layers = [embedder_nodes] * embedder_layers
        q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
    else:
        q1_embedder = eve_rl.network.component.ComponentDummy()

    q1_base = eve_rl.network.component.MLP(hidden_layers)
    q2_base = eve_rl.network.component.MLP(hidden_layers)
    policy_base = eve_rl.network.component.MLP(hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
    q1_optim = eve_rl.optim.Adam(
        q1,
        lr=lr,
    )
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
    q2_optim = eve_rl.optim.Adam(
        q2_base,
        lr=lr,
    )
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    policy = eve_rl.network.GaussianPolicy(
        policy_base, n_observations, n_actions, q1_embedder
    )
    policy_optim = eve_rl.optim.Adam(
        policy_base,
        lr=lr,
    )
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    sac_model = eve_rl.model.SACModel(
        lr_alpha=lr,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_optim,
        q2_optimizer=q2_optim,
        policy_optimizer=policy_optim,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
    )

    algo = eve_rl.algo.SAC(
        sac_model,
        n_actions=n_actions,
        gamma=gamma,
        reward_scaling=reward_scaling,
        stochastic_eval=stochastic_eval,
    )

    replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
        replay_buffer_size, batch_size, device_trainer
    )
    if not single:
        agent = eve_rl.agent.Synchron(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=device_trainer,
            worker_device=device_worker,
            n_worker=n_worker,
            normalize_actions=True,
            timeout_worker_after_reaching_limit=180,
        )
    else:
        agent = eve_rl.agent.Single(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device_trainer,
            normalize_actions=True,
        )

    return agent
