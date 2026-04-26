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

| Action                    | Commande                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| **clonner le projet**     | `git clone git@github.com:yahia-adam/deepRL_5iabd.git && cd deepRL_5iabd`|
| **Installer les dépendances** | `uv sync`                                                            |
| **Activer l'environement** | `source .venv/bin/activate`                                             |
| **Lancer un script**      | `uv run python -m mypythonlib.envs.quarto`                               |

---

## Configuration

Les dépendances sont gérées via **uv** et définies dans `pyproject.toml`.


Environnements de départ :
- pour tests : Line World
- pour tests : Grid World
- pour tests : TicTacToe versus Random
- Quarto (vs Random ou Heuristique)

Types d'agents à étudier :
- Random
- TabularQLearning (quand possible)
- DeepQLearning
- DoubleDeepQLearning
- DoubleDeepQLearningWithExperienceReplay
- DoubleDeepQLearningWithPrioritizedExperienceReplay
- REINFORCE
- REINFORCE with mean baseline
- REINFORCE with Baseline Learned by a Critic
- PPO A2C style
- RandomRollout
- Monte Carlo Tree Search (UCT)
- Expert Apprentice
- Alpha Zero
- MuZero
- MuZero stochastique

Métriques à obtenir (attention métriques pour la policy obtenue, pas pour la policy en mode entrainement)
:
- Score moyen (pour chaque agent) au bout de 1000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 10 000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 100 000 parties d'entrainement
- Score moyen (pour chaque agent) au bout de 1 000 000 parties d'entrainement (si possible)
- Score moyen (pour chaque agent) au bout de XXX parties d'entrainement (si possible)

- Temps moyen mis pour exécuter un coup

Si la partie est de durée variable :
- Longueur moyenne (nombre de step) d'une partie au bout de 1000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 10 000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 100 000 parties d'entrainement
- Longueur moyenne (nombre de step) d'une partie au bout de 1 000 000 parties d'entrainement (si possible)
- Longueur moyenne d'une partie au bout de XXX parties (si possible)

Il sera également nécessaire de présenter une interface graphique permettant de regarder jouer chaque
agent et également de mettre à disposition un agent 'humain'.
Pour chaque environnement et chaque algorithme, les étudiants devront étudier les performances de
l'algorithme et retranscrire leur résultats.