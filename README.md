# Deep RL Project

Projet d'implémentation d'algorithmes de Deep Reinforcement Learning et d'environnements de test.

## Structure du Projet

### Code Source

* **`src/agents/`** : Algorithmes de DRL (DQN, Double DQN, PPO, REINFORCE, MCTS, AlphaZero, MuZero, Tabular Q-Learning)
* **`src/envs/`** : Environnements de simulation (LineWorld, GridWorld, TicTacToe, Quarto)
* **`src/training/`** : Scripts d'entraînement

### Assets & Résultats

* **`game_assets/`** : Ressources visuelles des environnements (images, rendus graphiques)
* **`experimentation/`** : Résultats des expérimentations (logs TensorBoard)

---

## Commandes Utiles

| Action                    | Commande                    |
|---------------------------|-----------------------------|
| **Installer les dépendances** | `uv sync`               |
| **Lancer un script**      | `uv run python -m mypythonlib.envs.quarto`|

---

## Configuration

Les dépendances sont gérées via **uv** et définies dans `pyproject.toml`.