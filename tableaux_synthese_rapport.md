# Fiches Synthèse — Agents Deep RL

---

## 1. Q-Learning

### Comment ça marche

Table Q de taille `(num_states × num_actions)`, mise à jour par TD(0) :

```python
# q_learning.py
Q = np.zeros((num_states, num_actions))

td_target = final_reward + gamma * np.max(Q[new_state, :]) * (not done)
td_error  = td_target - Q[state, action]
Q[state, action] += current_lr * td_error
```

Le masquage des actions invalides se fait en copiant la ligne Q avec `-inf` :

```python
q_masked = np.full(num_actions, -np.inf)
q_masked[valid_actions] = Q[state, valid_actions]
action = np.argmax(q_masked)
```

**Hyperparamètres** : lr=0.1, γ=0.9, ε décroît de 1.0 → 0.0 (–0.00005/step)

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Espace d'états discret requis | Table indexée par entier | Inutilisable sur Quarto (état continu) |
| Convergence lente sur TicTacToe | Peu de mises à jour utiles en début d'ε-greedy | 62.1% @1k → 84.4% @100k |
| Optimal sur envs simples | TD(0) suffisant pour MDPs petits | 100% LineWorld/GridWorld dès @1k |

---

## 2. DQN

### Comment ça marche

Réseau `QNetwork` : `Linear(in,128) → ReLU → Linear(128,128) → ReLU → Linear(128,out)`

Le point clé de cette implémentation : **pas de réseau cible (target network)**. Le même réseau `q_net` est utilisé à la fois pour prédire Q(s,a) et calculer la cible :

```python
# dqn.py — _td_update
q_values  = q_net(state_t)[action]          # prédiction
q_next    = q_net(next_state_t).detach()    # cible (même réseau !)
td_target = reward + gamma * q_next.max() * (1 - done)
loss = F.mse_loss(q_values, td_target)
```

Masquage : avant argmax, les actions invalides reçoivent `-inf` sur la sortie du réseau.

**Hyperparamètres** : lr=1e-3, RMSprop(momentum=0.95), ε × 0.9995/épisode

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Instabilité d'entraînement | Pas de target network → cible qui bouge | Résultats non disponibles (instabilité trop forte) |
| Boucle infinie sur GridWorld | Sans target net, Q diverge et l'agent tourne en rond | draw_rate=100%, steps=10000 @10k |
| Corrélation temporelle | Pas de replay buffer → mises à jour sur transitions consécutives | Gradient biaisé, oscillations |

---

## 3. DDQN

### Comment ça marche

Deux réseaux : `q_net` (actif) + `target_net` (gelé, synchronisé toutes les 100 steps).  
L'astuce Double DQN : séparer la **sélection** de l'action (via `q_net`) de son **évaluation** (via `target_net`) :

```python
# ddqn.py — _ddqn_update
best_next_action = q_net(x_next).argmax(dim=1)
target_q_next    = target_net(x_next).gather(1, best_next_action.unsqueeze(1))
td_target        = reward + gamma * target_q_next * (1 - done)
```

Synchronisation périodique :
```python
if global_step % target_update_freq == 0:
    target_net.load_state_dict(q_net.state_dict())
```

Epsilon linéaire : 1.0 → 0.1 sur les 50% premiers épisodes.

**Hyperparamètres** : lr=2.5e-4, target_update_freq=100, ε linéaire

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| 0% sur LineWorld et GridWorld | Sans replay buffer, les transitions corrélées empêchent la convergence | DDQN @100k = 0% LW, 0% GW |
| Apprentissage très lent | Pas de replay → une seule transition par update | Besoin de beaucoup plus d'épisodes |
| Performances correctes sur TicTacToe | Environnement plus riche en signal | 79.3% @100k |
| ~50% sur Quarto | Espace d'action très grand, pas de mémorisation | 48.1% @100k ≈ aléatoire |

---

## 4. DDQN + Experience Replay

### Comment ça marche

Ajout d'un `ReplayBuffer` (deque de capacité 50k). L'agent stocke toutes les transitions et tire aléatoirement un batch :

```python
# ddqn_replay.py
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_prime, next_mask, done):
        self.buffer.append((s, a, r, s_prime, next_mask, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch   = [self.buffer[i] for i in indices]
        s, a, r, s_prime, next_mask, done = zip(*batch)
        return np.array(s), np.array(a), ...
```

L'entraînement démarre après `learning_starts=1000` transitions, toutes les `train_freq=4` steps.

**Hyperparamètres** : buffer=50k, batch=32, learning_starts=1000, train_freq=4, target_update=500

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| 100% sur LineWorld | Replay brise la corrélation temporelle | Convergence rapide dès @1k |
| Résultats manquants GridWorld+ | Temps d'entraînement trop long pour le projet | — dans les tableaux |
| Échantillonnage uniforme | Toutes les transitions ont la même priorité | Transitions rares peu réutilisées |

---

## 5. DDQN + PER (Prioritized Experience Replay)

### Comment ça marche

Remplacement du replay uniforme par un **SumTree** : les transitions sont échantillonnées proportionnellement à `|δ|^α` (priorité = erreur TD absolue).

```python
# ddqn_per.py
def push(self, *data):
    self.tree.add(self.max_priority ** self.alpha, data)

def update_priorities(self, indices, td_errors):
    priorities = np.abs(td_errors) + self.per_eps   # per_eps=1e-6
    for idx, p in zip(indices, priorities):
        self.tree.update(idx, float(p) ** self.alpha)
```

Pour corriger le biais d'échantillonnage, on applique des **poids d'importance sampling** β (annealé de 0.4 → 1.0) :

```python
weights = (self.tree.n_entries * sampling_probs) ** (-beta)
weights /= weights.max()   # normalisation
```

**Hyperparamètres** : α=0.6, β_start=0.4, β_end=1.0, per_eps=1e-6

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| 100% sur LineWorld (= Replay) | Env trop simple pour voir le gain | Idem DDQN+Replay |
| Overhead de calcul | SumTree update à chaque batch | Entraînement plus lent |
| β trop bas en début | Correction IS incomplète → biais résiduel | Convergence légèrement différée |

---

## 6. REINFORCE

### Comment ça marche

Algorithme de policy gradient Monte Carlo : on joue un épisode complet, puis on calcule les **returns** Gₜ en remontant le temps :

```python
# reinforce.py
def compute_returns(rewards, gamma):
    returns = []
    for t in range(len(rewards)):
        G_t = 0
        for power, r in enumerate(rewards[t:]):
            G_t += (gamma ** power) * r
        returns.append(G_t)
    return returns
```

La perte de politique utilise le gradient ∇ log π(aₜ|sₜ) × Aₜ. Trois variantes de baseline :

```python
# reinforce.py — compute_policy_loss
if critic_values is not None:                              # baseline = critic
    advantage = return_t - critic_value.detach()
    loss += -log_prob_t * advantage
else:
    advantage = return_t - baseline                        # 0 ou mean(returns)
    loss += -log_prob_t * advantage
```

**Hyperparamètres** : lr=0.001, γ=0.99, Adam, critic entraîné séparément (MSE)

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Haute variance | Returns Monte Carlo sans baseline | Lent sur TicTacToe @1k (61.3%) |
| Convergence finale excellente | Policy gradient sans biais | 100% TicTacToe @100k |
| Quarto ~50% | Espace énorme, signal rare → REINFORCE ne converge pas | 48.8% ≈ aléatoire |
| Critic baseline légère amélioration | Réduction de variance partielle | 52.6% avec critic vs 48.8% sans |

---

## 7. PPO

### Comment ça marche

PPO améliore REINFORCE en deux points : les **avantages GAE** (λ=0.95) et la **perte clippée** pour limiter les mises à jour trop grandes.

```python
# ppo.py — compute_gae
gae = 0.0
for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
    gae   = delta + gamma * lam * (1 - dones[t]) * gae
    advantages.insert(0, gae)
```

La perte clippée empêche le ratio π_new/π_old de trop s'éloigner de 1 :

```python
# ppo.py — compute_ppo_loss
ratio  = torch.exp(new_log_probs - old_log_probs)
surr1  = ratio * advantages
surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
return -torch.min(surr1, surr2).mean()
```

Le rollout de 16 steps est réutilisé 4 fois (update_epochs=4). Gradient clipping à 0.5 :

```python
nn.utils.clip_grad_norm_(actor_agent.parameters(), max_grad_norm)
```

**Hyperparamètres** : rollout_size=16, update_epochs=4, clip_eps=0.2, lam=0.95, max_grad_norm=0.5

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Rollout court (16 steps) | Crédit partiel, pas assez de contexte | PPO Quarto loss=1.53 (divergence) |
| Quarto ~50% | Même que REINFORCE, espace trop grand | 49.8% @100k |
| Excellents sur envs simples | GAE + clipping = stabilité forte | 100% LineWorld/GridWorld dès @1k |
| Loss Quarto diverge | Rollout trop court pour des épisodes de ~20 steps | PPO loss finale = 1.5292 |

---

## 8. MCRR (Monte Carlo Random Rollout)

### Comment ça marche

Pas d'entraînement. À chaque décision, on **simule** un budget fixe de parties aléatoires pour chaque action légale, puis on choisit celle avec la meilleure récompense moyenne :

```python
# mcrr.py
a_resource = num_simulations // len(valid_actions)   # budget uniforme par action
for test_action in valid_actions:
    for _ in range(a_resource):
        new_env = env.determinize()                   # clone de l'état courant
        _, reward, terminated, truncated, _ = new_env.step(test_action)
        while not (terminated or truncated):
            a = new_env.action_space.sample(mask=new_env.get_action_mask())
            _, reward, terminated, truncated, _ = new_env.step(a)
        total_reward += reward
    action_mean_rewards[test_action] = total_reward / a_resource
```

`env.determinize()` clone l'environnement pour ne pas modifier l'état réel.

**Hyperparamètres** : num_simulations=100 par décision

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Très lent à l'inférence | 100 parties simulées à chaque coup | 14-117 ms/move selon env |
| Budget uniforme sous-optimal | Actions rares reçoivent autant de sims que les bonnes | Légèrement inférieur à MCTS |
| Excellent sur Quarto | Simulation profonde suffit dans ce domaine | 94.6% vs ~50% des agents apprenants |
| GridWorld lent | Plus de steps par rollout → 117 ms/move | Mais 100% de win rate |

---

## 9. MCTS (Monte Carlo Tree Search / UCT)

### Comment ça marche

MCTS construit un arbre de recherche avec la formule UCB pour équilibrer exploration/exploitation. La nouveauté par rapport à MCRR : les estimations sont **améliorées** au fil des simulations et les actions de l'adversaire sont inversées :

```python
# mcts.py — best_child
exploit = child.value / child.visits
if child.player != self.player:
    exploit = -exploit           # adversaire : on minimise son score
explore = c_param * math.sqrt(math.log(n) / child.visits)
scores.append(exploit + explore)
```

Boucle principale : `sélection → expansion (env.determinize()) → simulation → rétropropagation`

**Hyperparamètres** : num_simulations=100, c_param=1.4

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Lent (similaire à MCRR) | Arbre à construire à chaque coup | 2.85–118 ms/move |
| Meilleur que MCRR sur tous les envs | UCB guide l'exploration mieux que le budget uniforme | 96.5% Quarto vs 94.6% MCRR |
| Peu efficace sur envs simples | MCRR déjà optimal → pas de gain | 100% des deux sur LineWorld |

---

## 10. Expert Apprenti

### Comment ça marche

Pipeline en deux temps :
1. **Collecte** : MCRR joue 10k épisodes, à chaque step on calcule les Q-values MCRR (50 sims) → dataset de paires `(état, q_values_mcrr)`
2. **Supervision** : un réseau de neurones apprend à imiter ces Q-values via MSE

```python
# expert_apprenti.py — collecte
for ep in range(num_episodes):
    while not done:
        q_vals = mcrr_q_values(env, num_simulations=50)   # Q-values expert
        dataset.append((state, q_vals))
        action = np.argmax(q_vals)
        state, reward, done, ... = env.step(action)

# entraînement
loss_fn    = nn.MSELoss()
optimizer  = Adam(model.parameters(), lr=1e-3)
for epoch in range(1000):
    for batch in DataLoader(dataset, batch_size=2048):
        pred = model(states)
        loss = loss_fn(pred, targets)
```

**Hyperparamètres** : 10k épisodes de collecte, 50 sims MCRR, batch=2048, 1000 epochs, lr=1e-3

### Problèmes rencontrés

| Problème | Cause | Effet mesuré |
|---|---|---|
| Quarto limité à 70.1% | Distribution shift : état à l'inférence ≠ dataset (l'adversaire n'est pas MCRR) | Vs 94.6% MCRR direct |
| TicTacToe 87.8% @ep1000 | Même problème + bruit dans les Q-values MCRR (50 sims) | Vs 87.7% MCRR |
| Rapide à l'inférence | Juste un forward pass réseau | 0.09–0.17 ms/move |
| Amélioration progressive | Plus d'épisodes = meilleur dataset | 58.1% →63.5% → 70.1% sur Quarto |

---

## Tableau de synthèse global

| Agent | LineWorld | GridWorld | TicTacToe | Quarto | Vitesse inférence |
|---|---|---|---|---|---|
| **Q-Learning** | 100% | 100% | 84.4% | ✗ | ~0.05 ms |
| **DQN** | instable | instable | instable | instable | ~0.5 ms |
| **DDQN** | 0% | 0% | 79.3% | 48.1% | ~0.5 ms |
| **DDQN+Replay** | 100% | — | — | — | ~0.5 ms |
| **DDQN+PER** | 100% | — | — | — | ~0.5 ms |
| **REINFORCE** | 100% | 100% | 100% | 48.8% | ~0.5 ms |
| **REINFORCE+mean** | 100% | 100% | 100% | 49.0% | ~0.5 ms |
| **REINFORCE+critic** | 100% | 100% | 100% | **52.6%** | ~0.5 ms |
| **PPO** | 100% | 100% | 100% | 49.8% | ~0.5 ms |
| **MCRR** | 100% | 100% | 87.7% | 94.6% | 14–117 ms |
| **MCTS** | 100% | 100% | **92.4%** | **96.5%** | 3–118 ms |
| **Expert Apprenti** | 100% | 100% | 87.8% | 70.1% | ~0.1 ms |

**Constat principal** : les agents apprenants (REINFORCE, PPO) convergent parfaitement sur les environnements à faible espace d'action mais échouent tous sur Quarto (~50% = niveau aléatoire). Seuls les agents de recherche (MCTS/MCRR) surmontent ce problème grâce à la simulation directe, au prix d'une inférence 30–200× plus lente.
