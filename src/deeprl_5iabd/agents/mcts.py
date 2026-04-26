import time
import math
import numpy as np
from gymnasium import Env

from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import plot_metric
from gymnasium.wrappers import RecordVideo


class MCTSNode:
    def __init__(self, env, parent=None, action=None,
                 terminal=False, terminal_reward=0.0):
        self.env = env
        self.parent = parent
        self.action = action

        self.children = []
        self.visits = 0
        self.value = 0.0

        self.terminal = terminal
        self.terminal_reward = terminal_reward

        if terminal:
            self.untried_actions = []
        else:
            mask = env.get_action_mask()
            self.untried_actions = list(np.where(mask == 1)[0])

    def is_terminal(self):
        return self.terminal

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def _is_max_node(self):
        # single-player → toujours en max
        if not getattr(self.env, "is_multi_player", False):
            return True
        return self.env.current_player == self.env.agent_player

    def best_child(self, c_param=1.4):
        is_max = self._is_max_node()
        log_n = math.log(self.visits) if self.visits > 0 else 0.0

        scores = []
        for child in self.children:
            exploit = child.value / child.visits
            if not is_max:
                exploit = -exploit
            explore = c_param * math.sqrt(log_n / child.visits)
            scores.append(exploit + explore)

        return self.children[int(np.argmax(scores))]

    # SELECTION
    def selection(self, c_param=1.4):
        node = self
        while (not node.is_terminal()
               and node.is_fully_expanded()
               and node.children):
            node = node.best_child(c_param)
        return node

    # EXPANSION
    def expansion(self):
        action = int(self.untried_actions.pop())

        new_env = self.env.determinize()
        _, reward, terminated, truncated, _ = new_env.step(action)
        terminal = bool(terminated or truncated)

        child = MCTSNode(new_env, parent=self, action=action,
                         terminal=terminal, terminal_reward=reward)
        self.children.append(child)
        return child

    # SIMULATION
    def simulation(self):
        if self.terminal:
            return self.terminal_reward

        env = self.env.determinize()
        total = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = env.get_action_mask()
            if mask.sum() == 0:
                break
            action = env.action_space.sample(mask=mask)
            _, reward, terminated, truncated, _ = env.step(action)
            total += reward

        return total

    # BACKPROPAGATION
    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)


def mcts(env, num_simulations=100):
    root = MCTSNode(env)

    if not root.untried_actions and not root.children:
        return None

    for _ in range(num_simulations):
        node = root.selection()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expansion()

        reward = node.simulation()
        node.backpropagate(reward)

    best = max(root.children, key=lambda c: c.visits)
    return best.action


def run_mcts(env: Env, num_episodes: int = 1_000, num_simulations: int = 50):
    print(f"starting mcts for {env.unwrapped} "
          f"num_episodes={num_episodes} num_simulations={num_simulations}")

    rewards_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)
    game_times_history = np.zeros(num_episodes + 1)

    is_multi = getattr(env, "is_multi_player", False)

    for episode in range(1, num_episodes+1):
        env.reset()
        terminated = False
        truncated = False
        reward = 0.0
        n_step = 0

        t0 = time.perf_counter()
        while not (terminated or truncated):
            if is_multi and env.current_player != env.agent_player:
                mask = env.get_action_mask()
                a = env.action_space.sample(mask=mask)
            else:
                a = mcts(env, num_simulations)

            _, reward, terminated, truncated, _ = env.step(a)
            n_step += 1

        game_time = time.perf_counter() - t0
        rewards_history[episode - 1] = reward
        nbr_steps_history[episode - 1] = n_step
        game_times_history[episode - 1] = game_time

        if episode % 100 == 0:
            print(f"episode {episode} reward={reward} n_step={n_step} game_time={game_time}s")

    save_dir = f"{settings.evaluation_logs_dir}/mcts/{env.unwrapped}"
    exp_name = f"mcts{num_simulations}_env_{env.unwrapped}"

    winrate_path = plot_metric(
        values=rewards_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="winrate",
    )
    nbr_steps_path = plot_metric(
        values=nbr_steps_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="nbr_steps",
    )
    game_time_path = plot_metric(
        values=game_times_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="game_time",
    )

    print(f"winrate_path={winrate_path}")
    print(f"nbr_steps_path={nbr_steps_path}")
    print(f"game_time_path={game_time_path}")


def hvr(env, num_episodes, num_simulations):
    for _ in range(num_episodes):
        env.reset()
        done = False
        reward = 0.0
        while not done:
            env.render()
            if env.current_player == env.agent_player:
                a = mcts(env, num_simulations)
            else:
                mask = env.get_action_mask()
                a = env._wait_for_human_click(mask)
            _, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        print(reward)
        env.render()
        time.sleep(3)
    env.close()

if __name__ == "__main__":
    # env = QuartoEnv(render_mode="human")
    # hvr(env, num_episodes=10, num_simulations=200)



    env = LineWorldEnv(render_mode="rgb_array")
    # env = GridWorldEnv(render_mode="rgb_array")
    # env = TicTacToeEnv(render_mode="rgb_array")
    # env = QuartoEnv(render_mode="rgb_array")

    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/mcts/{env.unwrapped}/eval/",
        episode_trigger=lambda ep: ep % 1_000 == 0,
    )
    video_env.state_id = env.state_id
    video_env.determinize = env.determinize
    video_env.get_action_mask = env.get_action_mask
    video_env.agent_player = env.agent_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, 'current_player', v)
    )
    run_mcts(video_env, num_episodes=10_000, num_simulations=100)
    video_env.close()
