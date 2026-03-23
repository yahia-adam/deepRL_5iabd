from time import perf_counter
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.tracking.tb_logger import TensorBoardLogger

def count_n_match_time(env: BaseEnv, num_episode):
    player = RandomPlayer(action_dim=len(env.get_action_space()))
    s = perf_counter()
    
    i = 0
    while (i <= num_episode):
        while (env.is_game_over()):
            action_spaces = env.get_action_space()
            a = player.forward(x=None, mask=action_spaces)
            env.step(a)
        i += 1
    e = perf_counter()

    total_time = e - s
    match_per_s = num_episode / total_time
    return total_time, match_per_s

if __name__ == "__main__":

    envs = [LineWorld(), GridWorld(), TicTacToe(), QuartoEnv()]
    total_match = 1_000_000

    print(f"\n\nTEST SUR {total_match} PARTIES ...")
    print("-" * 44)
    print(f"{'Environnement':<20} {'Durée (s)':>10} {'Matchs/sec':>12}")
    print("-" * 44)
    for env in envs:
        total_duration, match_per_s = count_n_match_time(env, total_match)
        print(f"{env.env_name:<20} {round(total_duration, 2):>10} {match_per_s:>12.0f}")