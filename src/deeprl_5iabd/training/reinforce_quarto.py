
import copy
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.tracking.tb_logger import TensorBoardLogger

def train_self_play_loop(iterations=10, episodes_per_gen=1000):
    env = QuartoEnv()
    input_size = len(env.get_observation_space())
    output_size = len(env.get_action_space())

    current_agent = PolicyNetwork(name="Agent_Gen_0", input_size=input_size, output_size=output_size)
    opponent = RandomPlayer(action_dim=output_size)

    print(f"Démarrage de l'auto-apprentissage : {iterations} générations.")
    for i in range(1, iterations + 1):
        gen_name = f"Gen_{i}"
        experiment_name = f"Quarto_SelfPlay_{gen_name}_vs_{opponent.name}"
        
        print(f"\n--- Entraînement {gen_name} ---")

        logger = TensorBoardLogger(
            log_dir=settings.training_logs_path, 
            experiment_name=experiment_name
        )

        current_agent = reinforce(
            env=env, 
            opponent_model=opponent, 
            reinforce_agent=current_agent, 
            logger=logger, 
            num_episodes=episodes_per_gen,
            early_stop=True
        )

        save_path = f"{experiment_name}.pth"
        current_agent.save(save_path)
        print(f"Modèle sauvegardé : {save_path}")
        opponent = current_agent.clone(name=f"Opponent_Gen_{i}")
        opponent.name = gen_name
        current_agent.name = f"Agent_Gen_{i}"

def train_one_model():
    env = QuartoEnv()
    experiment_name="Quarto_REINFORCE_vs_Random"
    model  = PolicyNetwork(name="Quarto_REINFORCE", input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    opponent_model = RandomPlayer(action_dim=len(env.get_action_space()))
    logger = TensorBoardLogger(
        log_dir=settings.training_logs_path, 
        experiment_name=experiment_name
    )

    model = reinforce(env=env, opponent_model=opponent_model, reinforce_agent=model, logger=logger, early_stop=True, early_stop_val=0.85)
    model.save(experiment_name+".pth")

if __name__ == "__main__":
    # train_one_model()
    train_self_play_loop(iterations=5, episodes_per_gen=2000)
