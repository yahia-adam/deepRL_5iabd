# Rapport d'analyse comparative — Agents de Deep Reinforcement Learning

**Auteurs :** Adam — Ayman — Tom  
**Cours :** Deep Reinforcement Learning — 5IABD  
**Date :** 29 / 04 / 2026  

---

## Table des matières

1. [Introduction et objectifs](#1-introduction-et-objectifs)
2. [Environnements étudiés](#2-environnements-étudiés)
3. [Protocole expérimental](#3-protocole-expérimental)
4. [Résultats globaux](#4-résultats-globaux)
5. [Analyse par environnement](#5-analyse-par-environnement)
   - 5.1 [LineWorld](#51-lineworld)
   - 5.2 [GridWorld](#52-gridworld)
   - 5.3 [TicTacToe](#53-tictactoe)
   - 5.4 [Quarto](#54-quarto)
6. [Analyse du coût computationnel](#6-analyse-du-coût-computationnel)
7. [Synthèse et discussion](#7-synthèse-et-discussion)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction et objectifs

Dans ce projet on a implémenté from scratch un ensemble d'algorithmes de deep reinforcement learning, et on les a testés sur 4 environnements de complexité croissante. L'idée c'est de pas juste appliquer des bibliothèques toutes faites, mais de vraiment comprendre comment chaque algorithme fonctionne et pourquoi il performe mieux ou moins bien selon le problème.

Les grandes familles d'algorithmes qu'on a comparées sont :
- Le **Q-Learning tabulaire**, comme baseline "classique"
- La **famille DQN** (DQN, DDQN, DDQN+Replay, DDQN+PER) qui ajoute progressivement des mécanismes de stabilisation
- Les **méthodes policy-gradient** (REINFORCE avec 3 variantes de baseline, PPO)
- Les **méthodes de planification** (MCRR, MCTS) qui utilisent un modèle de l'environnement
- L'**Expert Apprenti** qui combine planification et apprentissage par imitation

L'objectif est de comprendre dans quelles conditions chaque famille d'algorithme excelle ou échoue, et d'être capables d'expliquer les différences de performance qu'on observe.

---

## 2. Environnements étudiés

On a utilisé 4 environnements qui couvrent un spectre de complexité croissante.

**LineWorld** est le plus simple : une ligne de 5 cases, l'agent part du centre et doit aller à droite (+1) sans tomber dans le piège à gauche (-1). La solution optimale c'est littéralement "toujours aller à droite". C'est un test de sanité — si un algorithme échoue ici, c'est qu'il y a un bug.

**GridWorld** c'est une grille 5×5, l'agent part du coin haut-gauche et doit atteindre le coin bas-droit en évitant le piège en haut à droite. Légèrement plus complexe avec 25 états et 4 actions, mais ça reste très facile.

**TicTacToe** c'est le premier vrai défi. L'agent joue contre un adversaire aléatoire sur un plateau 3×3 et ne reçoit la récompense qu'à la fin (+1 victoire, -1 défaite, 0 nul). Il faut gérer le crédit assignment (attribuer la récompense finale aux bons coups), le masquage des actions illégales (cases déjà occupées), et l'aspect multi-joueur.

**Quarto** est de loin le plus difficile. C'est un jeu de stratégie sur un plateau 4×4 avec 16 pièces uniques à 4 attributs binaires. L'observation fait 166 dimensions, le jeu alterne entre 2 phases (choisir une pièce pour l'adversaire, puis placer la pièce reçue), et la condition de victoire (4 pièces alignées partageant un attribut commun) est complexe à appréhender pour un agent. C'est l'environnement qui va vraiment discriminer les algorithmes.

| Environnement | Espace obs | Actions | Type | Difficulté |
|---|---|---|---|---|
| LineWorld | 5 | 2 | Solo, déterministe | Triviale |
| GridWorld | 25 | 4 | Solo, déterministe | Faible |
| TicTacToe | 10 | 9 (masqué) | Multi-joueur vs Random | Moyenne |
| Quarto | 166 | 16 (masqué) | Multi-joueur vs Random, 2 phases | Élevée |

---

## 3. Protocole expérimental

Chaque agent apprenant est entraîné sur **100 000 épisodes**, avec des évaluations à **1k, 10k et 100k épisodes**. Chaque évaluation porte sur 1 000 épisodes joués hors entraînement (epsilon = 0 ou politique déterministe).

Les méthodes de planification (MCTS, MCRR) ne s'entraînent pas — elles sont évaluées directement avec **100 simulations par coup**.

L'Expert Apprenti a un pipeline en deux étapes :
1. Un expert MCRR joue 10 000 épisodes (avec 100 simulations) et génère un dataset de paires (état, Q-values)
2. Un réseau est entraîné en supervisé (MSE) sur ce dataset pendant **1 000 epochs**, avec évaluation aux epochs 100, 500 et 1 000.

Toutes les expériences sont faites avec seed = 42. Q-Learning a été testé avec les seeds 42, 123 et 7 pour vérifier la robustesse.

Les métriques collectées par épisode sont : récompense finale (win/loss/draw), nombre de coups, temps par coup en ms. Les fichiers JSON contiennent les 1000 épisodes + un bloc `summary` avec les statistiques globales.

---

## 4. Résultats globaux

Le tableau suivant résume les win rates au checkpoint final (100k épisodes pour les agents apprenants, ou meilleur checkpoint pour Expert Apprenti).

| Agent | LineWorld | GridWorld | TicTacToe | Quarto |
|---|---|---|---|---|
| **Q-Learning** | **100.0%** | **100.0%** | 84.4% | N/A |
| **DQN** | 100.0%† | *0.0%* ⚠️ | 74.7% | 48.6% |
| **DDQN** | *0.0%* ⚠️ | *0.0%* ⚠️ | 79.3% | 48.1% |
| **DDQN + Replay** | **100.0%** | — | — | — |
| **DDQN + PER** | **100.0%** | — | — | — |
| **REINFORCE (no baseline)** | **100.0%** | **100.0%** | **100.0%** | 48.8% |
| **REINFORCE (mean baseline)** | **100.0%** | **100.0%** | **100.0%** | 49.0% |
| **REINFORCE (critic baseline)** | **100.0%** | **100.0%** | **100.0%** | **52.6%** |
| **PPO** | **100.0%** | **100.0%** | **100.0%** | 49.8% |
| **MCRR (100 sim)** | **100.0%** | **100.0%** | 87.7% | 94.6% |
| **MCTS (100 sim)** | **100.0%** | **100.0%** | 92.4% | **96.5%** |
| **Expert Apprenti (epoch 1000)** | **100.0%** | **100.0%** | 87.8% | 70.1% |

*† DQN LineWorld 100k : 100% en eval, mais instable en training (diverge puis reconverge). Voir §5.1.*  
*Q-Learning non testé sur Quarto (espace d'états infini). DDQN+Replay et DDQN+PER testés uniquement sur LineWorld.*  
*DQN@1k sur LineWorld : draw_rate=100% (agent coincé, steps=10000) — artefact de début d'entraînement.*

Le premier truc qui saute aux yeux c'est que sur les environnements simples (LineWorld, GridWorld) pratiquement tout le monde converge à 100%, donc ça ne discrimine pas. C'est sur TicTacToe et surtout Quarto que les vraies différences apparaissent.

---

## 5. Analyse par environnement

### 5.1 LineWorld

Sur LineWorld et GridWorld, les vraies surprises viennent de la famille DQN — tout le reste converge à 100%.

**DQN** affiche 100% sur LineWorld à 100k épisodes en évaluation, mais masque une instabilité sévère en training. Par fenêtres de 500 épisodes :

```
DQN LineWorld — win rate pendant l'entraînement (fenêtre 500 épisodes)

100% |
 84% |              ████████
 60% |      ████        ████        ████      ████
 40% |  ████              ████  ████    ████
 20% |                      ██████
  0% |████
     0k    10k    20k    30k    40k    50k    60k    70k    80k    90k   100k
```

L'agent monte à ~84% vers l'épisode 30k, s'effondre à 20%, puis oscille jusqu'à la fin. Sans target network, les cibles TD bougent à chaque step — le réseau "chasse ses propres cibles". La `policy_loss` explose à **464 000** en moyenne, confirmant la divergence. Sur **GridWorld**, DQN n'arrive jamais à apprendre : **0%** à tous les checkpoints, ou draw 100% avec steps=10 000 (l'agent tourne en boucle jusqu'au timeout, signe que la politique est totalement aléatoire à ce stade).

**DDQN** sans replay buffer : **0%** sur LineWorld et GridWorld à tous les checkpoints (loss_rate = 1.0). Même avec le target network, sans décorrélation des transitions l'entraînement ne converge pas. Sur TicTacToe par contre, le DDQN monte progressivement (52% → 59% → **79%**), ce qui montre que le target network aide mais que l'absence de replay reste un frein.

**DDQN + Replay** et **DDQN + PER** convergent proprement à 100% dès 1k épisodes sur LineWorld avec une `policy_loss` de 0.051 pour PER. Chaque ajout stabilise l'entraînement de manière mesurable.

---

### 5.2 GridWorld

Sur GridWorld, la plupart des agents convergent à 100% — à l'exception de DQN (0%) et DDQN (0%) qui n'arrivent jamais à apprendre, pour les mêmes raisons que sur LineWorld. Ce qui est intéressant c'est plutôt la **longueur des épisodes** parmi les agents qui réussissent :

| Agent | Mean steps | Commentaire |
|---|---|---|
| Q-Learning | 8.0 | Optimal (chemin minimum) |
| REINFORCE (toutes variantes) | 8.0 | Optimal |
| PPO | 8.0 | Optimal |
| Expert Apprenti | 8.0 | Optimal |
| MCTS (100 sim) | 14.4 | Sous-optimal — chemin plus long |
| MCRR (100 sim) | 20.8 | Très sous-optimal |

Les méthodes d'apprentissage ont "mémorisé" le chemin le plus court (8 pas = minimum pour aller du coin haut-gauche au coin bas-droit). MCTS prend 14 pas en moyenne et MCRR 21 pas — elles arrivent à destination mais en prenant des chemins détournés parce que la politique par rollouts aléatoires n'est pas mémorisée entre les décisions.

Q-Learning est aussi remarquable en **temps d'inférence** : 0.06 ms par coup contre 117 ms pour MCRR. Sur des environnements aussi simples, l'overhead des réseaux de neurones et des simulations Monte Carlo n'est vraiment pas justifié.

---

### 5.3 TicTacToe

TicTacToe c'est le premier environnement vraiment intéressant pour la comparaison. On a beaucoup plus de variation entre les algorithmes.

#### Q-Learning tabulaire

Q-Learning atteint **84.4%** (seed 42) à 100k épisodes. La convergence est progressive et on voit clairement trois phases :

```
Q-Learning TicTacToe — win rate aux checkpoints (seed 42)

100% |
 90% |                              ████
 80% |              ████████████████
 70% |
 60% |████████████
     1k              10k              100k
     62.1%           80.2%            84.4%
```

La progression est régulière mais s'essouffle : on gagne 18 points entre 1k et 10k, puis seulement 4 points entre 10k et 100k. La table Q a convergi pour la plupart des états, mais les derniers %, ce sont des situations rares ou des nuls inévitables contre un adversaire aléatoire qui joue parfois un coup par chance.

Sur les 3 seeds, les résultats sont cohérents : seed 42 → 84.4%, seed 123 → 86.2%, seed 7 → 87.0%. La variance inter-seeds est faible, ce qui montre que l'algorithme converge de manière robuste.

La **limite fondamentale** du Q-Learning c'est qu'il ne peut pas généraliser entre états similaires. Deux configurations de plateau très proches ont des entrées complètement indépendantes dans la table — aucun partage de paramètres.

#### Famille DQN sur TicTacToe

C'est la première évaluation complète de DQN et DDQN sur un jeu adversarial. Les résultats sont contrastés :

| Agent | @1k | @10k | @100k |
|---|---|---|---|
| DQN | 77.4% | 64.4% | 74.7% |
| DDQN | 52.3% | 59.5% | **79.3%** |

DQN présente un comportement instable : il commence bien (77% à 1k) puis régresse à 64% à 10k avant de remonter à 75%. C'est la signature de l'instabilité sans target network — les Q-values oscillent et la politique avec elles. DDQN est plus régulier dans sa progression (52% → 59% → 79%) grâce au target network qui stabilise les cibles.

Mais les deux restent bien en dessous des méthodes policy-gradient (100%) — le signal de récompense terminal sparse est difficile à propager efficacement via TD bootstrap sans replay buffer.

#### Méthodes policy-gradient (REINFORCE et PPO)

Toutes les variantes REINFORCE et PPO atteignent **100%** à 100k épisodes. Mais la vitesse de convergence varie :

```
REINFORCE et PPO sur TicTacToe — win rate aux checkpoints (seed 42)

100% |              ████████████████████ (toutes variantes)
 80% |
 70% |          ████ (critic) / PPO
 60% |      ████ (no_baseline)
 50% |  ████ (mean_baseline)
     1k              10k              100k
```

| Agent | @1k | @10k | @100k |
|---|---|---|---|
| REINFORCE (no baseline) | 61.3% | 100.0% | 100.0% |
| REINFORCE (mean baseline) | 53.1% | 100.0% | 100.0% |
| REINFORCE (critic baseline) | 69.8% | 100.0% | 100.0% |
| PPO | 70.8% | 99.7% | 100.0% |

Ce qui est un peu surprenant c'est que REINFORCE sans baseline converge aussi vite que le critic à partir de 10k épisodes. En fait TicTacToe contre un adversaire aléatoire est suffisamment simple que même avec une haute variance, le signal de victoire finit par propager correctement en 10k épisodes. Le critic aide surtout en début d'entraînement.

À 1k épisodes par contre, l'écart est visible : 70% pour critic/PPO contre 53% pour mean baseline. La baseline critic donne un meilleur avantage estimé dès le début parce qu'elle s'adapte à chaque état (V(s) ≠ constante), alors que la mean baseline soustrait la même valeur à tous les états de l'épisode.

Un truc intéressant avec la **mean baseline** : à 1k épisodes elle est la pire (53%), mais à 100k épisodes elle est au même niveau que tout le monde (100%). C'est le signe que la réduction de variance de la mean baseline est trop grossière en début d'entraînement — elle peut ralentir la convergence si la baseline est mal calibrée.

#### Méthodes de planification

**MCTS** obtient 92.4% et **MCRR** 87.7% sur TicTacToe. C'est les meilleures performances "instantanées" (sans entraînement) mais elles restent en dessous des 100% des méthodes qui ont appris sur 100k épisodes.

La raison c'est que 100 simulations sur un arbre de jeu TicTacToe — qui peut aller jusqu'à 9 coups avec 9! configurations possibles — c'est pas énorme. MCTS peut manquer des branches importantes. Les agents entraînés eux ont "mémorisé" la stratégie optimale contre un adversaire aléatoire.

MCTS > MCRR (+4.7 points) parce que l'arbre UCB concentre les simulations sur les coups les plus prometteurs, alors que MCRR distribue son budget équitablement même sur des coups clairement mauvais.

#### Expert Apprenti

L'Expert Apprenti progresse de **81.7%** (epoch 100) à **87.8%** (epoch 1000), ce qui est exactement le niveau de son expert MCRR (87.7%). C'est cohérent — par imitation pure on ne peut pas surpasser le professeur.

```
Expert Apprenti TicTacToe — progression par epoch

90% |                              ████
    |              ████████████████
85% |████████████
    epoch 100       epoch 500       epoch 1000
    81.7%           86.8%           87.8%
```

La loss de supervision (MSE entre Q-values du réseau et Q-values MCRR) décroît de 0.384 à 0.038 sur 1000 epochs, ce qui montre que le réseau apprend bien à reproduire le comportement de l'expert.

---

### 5.4 Quarto

C'est là que tout devient vraiment intéressant, parce que les performances divergent massivement selon les approches.

#### Le mur des méthodes d'apprentissage

Le résultat le plus frappant c'est que **REINFORCE (toutes variantes) et PPO stagnent tous autour de 48-53%** sur Quarto — soit à peine mieux que du hasard (50%).

| Agent | @1k | @10k | @100k |
|---|---|---|---|
| REINFORCE (no baseline) | 48.9% | 51.4% | **48.8%** |
| REINFORCE (mean baseline) | 51.6% | 48.3% | **49.0%** |
| REINFORCE (critic baseline) | 49.2% | 47.3% | **52.6%** |
| PPO | 48.3% | 50.2% | **49.8%** |

Il n'y a **aucune tendance à l'amélioration**. Les chiffres oscillent autour de 50% sans jamais montrer une progression claire du début à la fin de l'entraînement. C'est fondamentalement différent de TicTacToe où on voyait des courbes qui montaient.

```
Toutes méthodes d'apprentissage sur Quarto — win rate (très stable ~50%)

55% |                                                 ████ (critic)
50% |████████████████████████████████████████████████████ (no/mean/PPO)
45% |
     1k                    10k                    100k
```

Pourquoi est-ce que ça ne marche pas ? Il y a plusieurs raisons qui se cumulent :

**1. Le crédit assignment est extrêmement difficile.** En phase SELECT, l'agent choisit une pièce à donner à l'adversaire. Cette action peut permettre ou empêcher l'adversaire de gagner 4 à 6 tours plus tard. Avec des épisodes de 14 à 26 coups et γ = 0.99, le signal de récompense terminal est techniquement propagé, mais dans un espace de 166 dimensions, l'agent n'arrive pas à relier l'action SELECT au résultat final.

**2. L'espace d'état est gigantesque pour 100k épisodes.** Avec 16 pièces à 4 attributs, 16 positions sur le plateau et 2 phases, le nombre d'états distincts est astronomique. En 100k épisodes, on ne visite qu'une infime fraction de l'espace — la généralisation est insuffisante.

**3. La récompense est uniquement terminale.** Il n'y a aucun signal intermédiaire pour guider l'apprentissage. L'agent reçoit +1 ou -1 à la fin sans aucune indication de si ses coups intermédiaires étaient bons ou mauvais.

**4. La complexité des deux phases.** L'agent doit apprendre deux politiques distinctes (SELECT et PLACE) et les coordonner de manière cohérente.

Le seul résultat légèrement au-dessus des autres est **REINFORCE critic à 52.6%**. C'est marginal, mais le critic qui apprend V(s) aide légèrement à mieux évaluer les états. La différence avec les autres variantes reste cependant dans le bruit statistique sur 1000 épisodes.

Pour référence, même en entraînant plus longtemps, on ne s'attend pas à des améliorations drastiques avec ces approches — les méthodes qui marchent vraiment sur des jeux de complexité similaire (Chess, Go) utilisent du self-play pendant des millions de parties avec des architectures bien plus grandes.

#### L'Expert Apprenti : la vraie surprise

L'Expert Apprenti obtient **70.1%** après 1000 epochs, soit la meilleure performance parmi les agents d'apprentissage sur Quarto avec +17 points par rapport à PPO/REINFORCE.

```
Expert Apprenti Quarto — win rate par epoch

75% |                                                 ████
70% |                              ████████████████████
65% |              ████████████████
60% |████████████
     epoch 100       epoch 500       epoch 1000
     58.1%           63.5%           70.1%
```

La progression est constante et nette — pas de stagnation autour de 50% comme pour les méthodes model-free. Pourquoi est-ce que ça marche mieux ?

La différence fondamentale c'est la **qualité du signal d'apprentissage**. L'Expert Apprenti n'apprend pas d'un signal binaire +1/-1 en fin de partie. Il apprend par **supervision directe** sur des Q-values estimées par MCRR : pour chaque état, il y a une estimation numérique de la valeur de chaque action (via 50 simulations par action). Ces Q-values contiennent une information bien plus riche que le résultat terminal — elles encodent directement "l'action A est 1.5× meilleure que l'action B dans ce contexte précis".

La loss MSE décroît de 0.251 à 0.107 sur 1000 epochs (final_loss = 0.107, min_loss = 0.107 atteint presque à la fin), ce qui montre que le réseau apprend progressivement à reproduire le raisonnement de l'expert plutôt que juste son résultat.

**La limite principale de l'Expert Apprenti** c'est le **distribution shift** : l'expert génère des données depuis des états qu'il visite lui-même. Mais une fois qu'on déploie l'étudiant, il peut se retrouver dans des états que l'expert n'a jamais rencontrés — et là sa politique devient imprévisible. C'est pour ça qu'on plafonne à 70% et pas à 95%+ comme MCTS.

#### Les méthodes de planification dominent

**MCTS obtient 96.5%** et **MCRR 94.6%** — sans aucun entraînement, juste avec 100 simulations par coup. C'est sans comparaison les meilleures performances sur Quarto.

Ces méthodes ont un avantage fondamental : elles utilisent un **modèle de l'environnement** (accès à `env.determinize()` pour simuler des parties entières). Là où les méthodes model-free doivent estimer la valeur d'un état à partir d'une fonction apprise, MCTS et MCRR peuvent directement simuler ce qui va se passer et calculer le résultat attendu.

MCTS (96.5%) > MCRR (94.6%) grâce à la formule UCB qui concentre les 100 simulations sur les coups prometteurs. MCRR distribue son budget uniformément — avec 16 actions légales au maximum, ça fait 6 simulations par action en moyenne, ce qui est peu. MCTS peut allouer 40+ simulations à un coup prometteur et seulement 2 à un coup clairement mauvais.

Le prix à payer c'est le **temps d'inférence** : 35 ms par coup pour MCTS et 30 ms pour MCRR, contre 0.4-0.5 ms pour PPO/REINFORCE et seulement 0.09 ms pour l'Expert Apprenti. Dans un contexte de jeu en temps réel ou de déploiement en production, cette différence est rédhibitoire.

---

## 6. Analyse du coût computationnel

Le temps par coup est une métrique souvent négligée dans les comparaisons de RL, mais elle est cruciale pour les applications réelles.

```
Temps par coup (ms) — comparaison toutes méthodes

MCTS (Quarto)         ████████████████████████████████████ 35.1 ms
MCRR (Quarto)         ████████████████████████████████ 30.3 ms
MCTS (TicTacToe)      █████████████ 13.1 ms
MCRR (TicTacToe)      █████████████ 13.6 ms
DQN / REINFORCE / PPO ▌ 0.5-0.9 ms
Expert Apprenti       ▏ 0.09-0.10 ms
Q-Learning            ▏ 0.04-0.07 ms
```

| Agent | Quarto (ms) | TicTacToe (ms) | Ratio vs Expert |
|---|---|---|---|
| MCTS (100 sim) | 35.1 | 13.1 | ~390× |
| MCRR (100 sim) | 30.3 | 13.6 | ~337× |
| PPO / REINFORCE | 0.42-0.49 | 0.48-0.61 | ~5× |
| Expert Apprenti | **0.09** | **0.10** | 1× |
| Q-Learning | — | 0.07 | ~0.8× |

L'**Expert Apprenti est le plus rapide parmi les méthodes profondes** (0.09 ms) — plus rapide même que REINFORCE et PPO qui ont pourtant des réseaux moins larges. C'est parce que son réseau (3 couches de 128 neurones) est très compact et n'a pas de rollouts à calculer.

Le ratio MCTS/Expert Apprenti est de ~390× sur Quarto. Si on imagine un jeu en temps réel avec une contrainte de 10 ms par coup, MCTS est inutilisable (35 ms) mais l'Expert Apprenti est tranquillement dans le budget (0.09 ms).

**Rapport qualité/vitesse :** c'est clairement l'Expert Apprenti qui offre le meilleur compromis sur Quarto. 70.1% de win rate pour 0.09 ms par coup, c'est imbattable parmi les méthodes sans modèle.

---

## 7. Synthèse et discussion

### Ce qui confirme la théorie

**La progression DQN → DDQN → DDQN+Replay → DDQN+PER se justifie empiriquement.** Sur LineWorld, DQN diverge à 100k épisodes (0% de win rate) alors que DDQN+PER reste stable à 100%. Chaque amélioration algorithmique apporte quelque chose de concret :
- Target network → stabilise les cibles TD (DDQN vs DQN)
- Replay buffer → décorrèle les transitions (DDQN+Replay)
- Prioritisation → focalise l'apprentissage sur les transitions informatives (DDQN+PER)

**La baseline critic réduit bien la variance de REINFORCE.** À 1k épisodes sur TicTacToe, REINFORCE critic (69.8%) > PPO (70.8%) > REINFORCE no_baseline (61.3%) > REINFORCE mean (53.1%). L'ordre est quasi-conforme à ce que la théorie prédit pour la réduction de variance.

**MCTS > MCRR à budget égal.** Sur tous les environnements, MCTS fait mieux que MCRR avec le même budget de 100 simulations. La sélection UCB est clairement plus efficace que le budget uniforme de MCRR.

### Les surprises

**REINFORCE converge aussi vite que PPO sur TicTacToe.** On aurait attendu PPO significativement meilleur grâce au clipping et aux K epochs. En fait les deux arrivent à 100% dès 10k épisodes. TicTacToe contre un adversaire aléatoire est peut-être trop simple pour que les avantages de PPO soient visibles.

**Le DQN diverge sur LineWorld malgré 100k épisodes.** C'est le résultat le plus contre-intuitif du projet. On aurait pu penser que même sans replay ni target network, l'algorithme finirait par apprendre sur un environnement aussi trivial. Mais l'instabilité des updates TD sans target network provoque une divergence persistante et non résolue même après 100k épisodes.

**L'Expert Apprenti explose les méthodes RL classiques sur Quarto.** Un écart de 17 points (70.1% vs ~50%) c'est énorme. On ne s'attendait pas à ce que la différence entre apprendre par signal terminal sparse et apprendre par supervision directe soit aussi massive.

### Les limites de nos expériences

Il y a quelques points qui mériteraient d'être approfondis :

**DQN/DDQN/DDQN+Replay n'ont pas été évalués sur TicTacToe et Quarto.** On a surtout testé DDQN+PER sur LineWorld. Ça aurait été intéressant de voir si la famille DQN se comporte différemment des méthodes policy-gradient sur les jeux adversariaux.

**Un seul adversaire : Random.** Tous les agents apprennent contre un adversaire aléatoire fixe. Un adversaire plus fort (MCTS ou Expert Apprenti) changerait probablement la hiérarchie. C'est l'approche du self-play d'AlphaGo/AlphaZero.

**100k épisodes c'est peut-être insuffisant pour Quarto.** Les méthodes state-of-the-art sur des jeux de complexité similaire utilisent des millions de parties. Les 50% de REINFORCE/PPO reflètent peut-être un manque de données plus qu'une limite algorithmique fondamentale.

---

## 8. Conclusion

Voilà ce qu'on retient de cette étude.

**Sur les environnements simples** (LineWorld, GridWorld), tous les algorithmes convergent et les différences sont mineures. Q-Learning est parfaitement adapté — c'est l'algorithme le plus rapide (0.06 ms) et le plus simple à implémenter, et il donne des résultats optimaux. Il n'y a aucune raison d'utiliser un réseau de neurones ici.

**Sur TicTacToe**, les méthodes policy-gradient (REINFORCE, PPO) convergent plus vite et vers de meilleures performances que Q-Learning (100% vs 84%). PPO et REINFORCE critic convergent de manière similaire, avec un léger avantage au REINFORCE critic en début d'entraînement. Les méthodes de planification sont compétitives sans entraînement mais plafonnent à 92% avec 100 simulations.

**Sur Quarto**, le constat est sans appel : les méthodes model-free (REINFORCE, PPO) ne parviennent pas à apprendre quelque chose d'utile en 100k épisodes, stagnant autour de 50%. L'Expert Apprenti (70.1%) est le meilleur agent d'apprentissage, grâce à la supervision directe par Q-values de l'expert. MCTS (96.5%) domine mais est inutilisable en temps réel (35 ms/coup). L'Expert Apprenti offre le meilleur rapport performance/vitesse d'inférence (70.1% pour 0.09 ms).

**La leçon principale** c'est que la qualité du signal d'apprentissage est souvent plus importante que la sophistication de l'algorithme. REINFORCE avec un signal terminal sparse échoue là où l'Expert Apprenti avec un signal dense (Q-values) réussit. Et MCTS avec un accès au modèle de l'environnement dépasse tout le monde sans même "apprendre". Sur des problèmes difficiles, le bon inductive bias (modèle de l'environnement, signal dense, architecture adaptée) compte plus que l'algorithme lui-même.
