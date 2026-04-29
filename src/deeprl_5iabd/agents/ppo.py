import os
import json
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo

from deeprl_5iabd.helper import softmax_with_mask, plot_metric
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings


SEEDS = (42,)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActorAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_size),
        )

    def forward(self, state, action_mask):
        logits = self.network(state)
        probs = softmax_with_mask(logits, action_mask)
        return probs


class CriticAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, state_tensor):
        return self.network(state_tensor).squeeze(-1)


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_gae(
    rewards: list[float],
    values: list[float],
    next_value: float,
    dones: list[bool],
    gamma: float,
    lam: float,
) -> list[float]:

    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
    return advantages


def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_critic_loss(values, returns):
    return ((values - returns) ** 2).mean()


def opponent_step(env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


def ppo(
    env: gym.Env,
    num_episodes: int = 100_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    lam: float = 0.95,                # FIX #2 : paramètre GAE-λ
    clip_eps: float = 0.2,
    update_epochs: int = 4,
    rollout_size: int = 16,
    batch_size: int = 64,              # FIX #4 : mini-batch shufflé
    entropy_coef: float = 0.01,        # FIX #6 : entropy bonus
    max_grad_norm: float = 0.5,        # FIX #7 : gradient clipping
    checkpoints: tuple = (1_000, 10_000, 100_000),
    seed: int = 42,
):
    set_seed(seed)

    actor_agent = ActorAgent(env)
    critic_agent = CriticAgent(env)

    actor_optimizer = optim.Adam(actor_agent.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes + 1)
    loss_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)
    time_per_move_history = np.zeros(num_episodes + 1)

    agent_name = f"ppo_clip={clip_eps}_epochs={update_epochs}_rollout={rollout_size}"

    is_multi = getattr(env, "is_multi_player", False)

    # ─── Buffers de rollout ───────────────────────────────────────────────────
    buf_states: list[torch.Tensor] = []
    buf_actions: list[torch.Tensor] = []
    buf_masks: list[np.ndarray] = []
    buf_old_log_probs: list[torch.Tensor] = []
    buf_returns: list[float] = []
    buf_advantages: list[float] = []

    last_actor_loss = 0.0

    for episode in range(1, num_episodes + 1):

        # ── Collecte d'un épisode ─────────────────────────────────────────────
        ep_states: list[torch.Tensor] = []
        ep_actions: list[torch.Tensor] = []
        ep_masks: list[np.ndarray] = []
        ep_old_log_probs: list[torch.Tensor] = []
        ep_rewards: list[float] = []      # FIX #1 : vraies récompenses par step
        ep_values: list[float] = []
        ep_dones: list[bool] = []

        n_step = 0
        final_reward = 0.0

        state, _ = env.reset()
        done = False
        truncated = False

        episode_start = time.perf_counter()

        while not (done or truncated):
            if is_multi:
                while not (done or truncated) and env.current_player != env.agent_player:
                    state, final_reward, done, truncated, _ = opponent_step(env)
                    n_step += 1

            while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                action_mask = env.get_action_mask()
                state_tensor = torch.tensor(state).float()

                with torch.no_grad():
                    action_probs = actor_agent(state_tensor, action_mask)
                    value = critic_agent(state_tensor).item()   # FIX #2

                dist = Categorical(action_probs)
                action = dist.sample()

                ep_states.append(state_tensor)
                ep_actions.append(action)
                ep_masks.append(action_mask)
                ep_old_log_probs.append(dist.log_prob(action).detach())
                ep_values.append(value)

                state, final_reward, done, truncated, _ = env.step(action.item())
                n_step += 1

                # FIX #1 : on stocke la vraie récompense à chaque step
                ep_rewards.append(final_reward)
                ep_dones.append(done or truncated)

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[episode] = episode_time / max(n_step, 1)
        rewards_history[episode] = final_reward
        nbr_steps_history[episode] = n_step

        # ── Calcul des returns et avantages GAE ───────────────────────────────
        n = len(ep_states)
        if n > 0:
            # next_value = 0 si l'épisode est terminé (done), sinon bootstrap
            next_value = 0.0 if (done or truncated) else ep_values[-1]

            # FIX #2 : GAE
            ep_advantages = compute_gae(
                ep_rewards, ep_values, next_value, ep_dones, gamma, lam
            )
            # Returns = avantages + valeurs (targets du critic)
            ep_returns = [adv + val for adv, val in zip(ep_advantages, ep_values)]

            buf_states.extend(ep_states)
            buf_actions.extend(ep_actions)
            buf_masks.extend(ep_masks)
            buf_old_log_probs.extend(ep_old_log_probs)
            buf_returns.extend(ep_returns)
            buf_advantages.extend(ep_advantages)

        # ── Mise à jour PPO toutes les rollout_size épisodes ──────────────────
        if episode % rollout_size == 0 and len(buf_states) > 0:

            returns_tensor = torch.tensor(buf_returns, dtype=torch.float32)
            advantages_tensor = torch.tensor(buf_advantages, dtype=torch.float32)
            old_log_probs_tensor = torch.stack(buf_old_log_probs)
            actions_tensor = torch.stack(buf_actions)

            # FIX #4 : normalisation des avantages sur tout le buffer
            if len(advantages_tensor) > 1:
                advantages_tensor = (
                    (advantages_tensor - advantages_tensor.mean())
                    / (advantages_tensor.std() + 1e-8)
                )

            # FIX #4 : batchification + shuffle
            states_tensor = torch.stack(buf_states)          # [N, obs_size]
            masks_tensor = torch.tensor(
                np.array(buf_masks), dtype=torch.float32
            )                                                # [N, action_size]

            N = len(buf_states)

            for _ in range(update_epochs):
                # Shuffle à chaque epoch
                perm = torch.randperm(N)

                for start in range(0, N, batch_size):
                    idx = perm[start: start + batch_size]

                    s_batch = states_tensor[idx]
                    a_batch = actions_tensor[idx]
                    m_batch = masks_tensor[idx]
                    old_lp_batch = old_log_probs_tensor[idx]
                    adv_batch = advantages_tensor[idx]
                    ret_batch = returns_tensor[idx]

                    # FIX #4 : forward pass en batch (plus de boucle état-par-état)
                    probs = actor_agent(s_batch, m_batch)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(a_batch)
                    entropy = dist.entropy().mean()

                    values = critic_agent(s_batch)

                    actor_loss = compute_ppo_loss(
                        new_log_probs, old_lp_batch, adv_batch, clip_eps
                    )
                    # FIX #6 : entropy bonus
                    actor_loss = actor_loss - entropy_coef * entropy

                    critic_loss = compute_critic_loss(values, ret_batch)

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # FIX #7 : gradient clipping
                    nn.utils.clip_grad_norm_(actor_agent.parameters(), max_grad_norm)
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic_agent.parameters(), max_grad_norm)
                    critic_optimizer.step()

                    last_actor_loss = actor_loss.item()

            buf_states.clear()
            buf_actions.clear()
            buf_masks.clear()
            buf_old_log_probs.clear()
            buf_returns.clear()
            buf_advantages.clear()

        loss_history[episode] = last_actor_loss

        if episode % 100 == 0:
            recent_rewards = rewards_history[max(1, episode - 100):episode + 1]
            recent_tpm = time_per_move_history[max(1, episode - 100):episode + 1]
            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {episode} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Policy Loss={loss_history[episode]:.4f} | "
                f"Time/move={np.mean(recent_tpm) * 1000:.2f}ms"
            )

        if episode in checkpoints:
            model_dir = f"{settings.models_path}/ppo/{env.unwrapped}/seed_{seed}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/policy_{agent_name}_{episode}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(actor_agent.state_dict(), f)
            print(f"Model saved: {model_path}")

    # ── Métriques (identiques à l'original) ───────────────────────────────────
    plot_dir = f"{settings.training_logs_dir}/ppo/{env.unwrapped}/seed_{seed}/train"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=100,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=loss_history, save_dir=plot_dir, window_size=100,
                exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}", metric_name="loss")
    plot_metric(values=nbr_steps_history, save_dir=plot_dir, window_size=100,
                exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=100,
                exp_name=f"time_per_move_{agent_name}_env_{env.unwrapped}", metric_name="time_per_move")

    train_results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "seed": seed,
        "num_episodes": num_episodes,
        "hyperparameters": {
            "lr": lr,
            "gamma": gamma,
            "lam": lam,
            "clip_eps": clip_eps,
            "update_epochs": update_epochs,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
        },
        "summary": {
            "mean_reward": float(np.mean(rewards_history[1:])),
            "win_rate": float(np.mean(rewards_history[1:] == 1)),
            "loss_rate": float(np.mean(rewards_history[1:] == -1)),
            "draw_rate": float(np.mean(rewards_history[1:] == 0)),
            "mean_policy_loss": float(np.mean(loss_history[1:])),
            "mean_steps": float(np.mean(nbr_steps_history[1:])),
            "std_steps": float(np.std(nbr_steps_history[1:])),
            "mean_time_per_move_ms": float(np.mean(time_per_move_history[1:]) * 1000),
        },
        "episodes": [
            {
                "episode": int(ep),
                "reward": float(rewards_history[ep]),
                "policy_loss": float(loss_history[ep]),
                "n_steps": int(nbr_steps_history[ep]),
                "time_per_move_ms": float(time_per_move_history[ep] * 1000),
            }
            for ep in range(1, num_episodes + 1)
        ],
    }

    json_path = f"{plot_dir}/{agent_name}_{num_episodes}.json"
    with open(json_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"Training metrics saved: {json_path}")

    return actor_agent


def eval_agent(env, num_episodes=1_000, model_name="policy_ppo_clip=0.2_epochs=4_rollout=16_1000.pkl", seed: int = 42):
    agent = ActorAgent(env)
    with open(f"{settings.models_path}/ppo/{env.unwrapped}/seed_{seed}/{model_name}", "rb") as f:
        state_dict = pickle.load(f)
    agent.load_state_dict(state_dict)
    agent.eval()

    is_multi = getattr(env, "is_multi_player", False)

    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    for i in range(num_episodes):
        n_step = 0
        episode_start = time.perf_counter()

        with torch.no_grad():
            state, _ = env.reset()
            done = False
            truncated = False
            reward = 0.0

            while not (done or truncated):

                if is_multi:
                    while not (done or truncated) and env.current_player != env.agent_player:
                        state, reward, done, truncated, _ = opponent_step(env)
                        n_step += 1

                while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                    action_mask = env.get_action_mask()
                    state_tensor = torch.tensor(state).float()

                    action_probs = agent(state_tensor, action_mask)
                    dist = Categorical(action_probs)
                    action = dist.sample().item()

                    state, reward, done, truncated, _ = env.step(action)
                    n_step += 1

        episode_time = time.perf_counter() - episode_start
        rewards_history[i] = reward
        n_steps_history[i] = n_step
        time_per_move_history[i] = episode_time / max(n_step, 1)

    base = model_name.replace("policy_", "").replace(".pkl", "")
    agent_name, checkpoint_str = base.rsplit("_", 1)
    checkpoint = int(checkpoint_str)

    plot_dir = f"{settings.training_logs_dir}/ppo/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=n_steps_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="time_per_move")

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "num_episodes": num_episodes,
        "summary": {
            "mean_reward": float(np.mean(rewards_history)),
            "win_rate": float(np.mean(rewards_history == 1)),
            "loss_rate": float(np.mean(rewards_history == -1)),
            "draw_rate": float(np.mean(rewards_history == 0)),
            "mean_steps": float(np.mean(n_steps_history)),
            "std_steps": float(np.std(n_steps_history)),
            "mean_time_per_move_ms": float(np.mean(time_per_move_history) * 1000),
        },
        "episodes": [
            {
                "episode": int(i),
                "reward": float(rewards_history[i]),
                "n_steps": int(n_steps_history[i]),
                "time_per_move_ms": float(time_per_move_history[i] * 1000),
            }
            for i in range(num_episodes)
        ],
    }

    json_path = f"{plot_dir}/{agent_name}_{checkpoint}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(
        f"[EVAL {env.unwrapped}/{agent_name}@{checkpoint} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}"
    )
    return results


def train_for_env(env, seed, clip_eps=0.2, update_epochs=4, rollout_size=16):
    ppo(env, num_episodes=100_000, clip_eps=clip_eps, update_epochs=update_epochs,
        rollout_size=rollout_size, seed=seed)


def eval_all_models_for_env(env, seed, clip_eps=0.2, update_epochs=4, rollout_size=16):
    tag = f"ppo_clip={clip_eps}_epochs={update_epochs}_rollout={rollout_size}"
    for n in (1_000, 10_000, 100_000):
        eval_agent(env, num_episodes=1_000, model_name=f"policy_{tag}_{n}.pkl", seed=seed)


def wrap_video(env, mode, seed, episode_num_trigger):
    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/ppo/{env.unwrapped}/seed_{seed}/{mode}/",
        episode_trigger=lambda ep: ep % episode_num_trigger == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    video_env.agent_player = env.agent_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, 'current_player', v)
    )
    return video_env


if __name__ == "__main__":
    env_classes = [LineWorldEnv, GridWorldEnv, TicTacToeEnv, QuartoEnv]

    for seed in SEEDS:
        for EnvCls in env_classes:
            #  TRAIN
            env_train = EnvCls(render_mode="rgb_array")
            video_env_train = wrap_video(env_train, "train", seed, 10_000)
            print(f"\n{'=' * 60}\nTRAIN {env_train.unwrapped} | seed={seed}\n{'=' * 60}")
            train_for_env(video_env_train, seed)
            video_env_train.close()

            #  EVAL
            env_eval = EnvCls(render_mode="rgb_array")
            video_env_eval = wrap_video(env_eval, "eval", seed, 100)
            print(f"\n{'=' * 60}\nEVAL {env_eval.unwrapped} | seed={seed}\n{'=' * 60}")
            eval_all_models_for_env(video_env_eval, seed)
            video_env_eval.close()