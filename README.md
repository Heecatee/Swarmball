# Swarmball.ai
## Documentation
https://docs.google.com/document/d/1d17AlUiGnsbga3XQSN1vrIGNN-0AY6xPdZwqkYeZTHM/edit?usp=sharing
## Participants
* Maria Korkuć
* Wioletta Kurek
* Wojciech Korzybski
* Patryk Radoń (Team Leader)
* Karol Kocierz

## Short description of the idea (for a detailed plan please go [here](https://github.com/Heecatee/Swarmball/blob/master/detailed_plan.md))

Swarmball.ai is a project based on real-life  problem of adapting the robotic tool to the environment when the task is to simply move some objects around. We have noticed, that there may be a more **versatile** approach.

### Go nanobots!

We will be working on a simplified problem using only **two dimensions**. Instead of one robot, we will be using the **swarm of small balls** with an ability to move right or left. The environment will be a bumpy plane full of obstacles, generated randomly or created by an evil human user. Their task is defined as follows: *Move the object to the far right wall.*

### The hive mind

The steering is going to be fully automated with **artificial intelligence**, adapting to any environment you make it run on. It will not only be forced to make the balls move the object, but to do it as quickly as possible. This will be accomplished by chasing the ball with a **Death Ray**, which will increase in speed as the algorithm becomes more effective.


### Technologies used:
  -PyMunk, 
  -PyGame, 
  -OpenAI gym, 
  -PyTorch, 
  -Neural Networks, 
  -Reinforcement Learning
 
### What will we learn?

This project is mainly aimed at introducing everyone involved to the concepts of deep reinforcement learning, in an end-to-end machine learning project. First, we wil learn how to prepare our own environment from  scratch using python wrappers for phisics engines. After that, we will plan out the training using techniques such as a2c and from that we will dive into the  state-of-art PPO. All that will be done using PyTorch - a deep learning framework, currently leading in the field of deep learning research.
The next step will introduce us to a process of tuning and training the model in a fast sparse environments or using our own GPU boosted machines. All that will hopefully lead us to a working product, and most of all, every single one of us will be able to say that they truely made an Artificial Intelligence.

### Preview
![](https://i.imgur.com/waH6dxF.gif)


# SwarmBall Simulation — Technical Runbook

This README describes the technical workflow for building the Docker image, verifying the environment, running tests, training/evaluating PPO models, and displaying the simulation with graphical mapping.

---

## 1. Project Overview

The project contains a physics-based SwarmBall simulation integrated with reinforcement learning. The trained PPO agent controls cluster thresholds in order to push the goal object toward a target position before it is caught by the enemy.

Main components:

```text
environment/
  swarmball_env.py              # Gymnasium environment
  simulation/                   # PyGame/Pymunk simulation

scripts/
  check_env.py                  # Gymnasium/SB3 compatibility check
  train_ppo.py                  # PPO training script
  evaluate_random.py            # Random baseline evaluation
  evaluate_model.py             # Trained model evaluation
  watch_model.py                # Graphical model visualization
  plot_training_progress.py     # Training plots

runs/
  <run_name>/                   # Models, logs, checkpoints

plots/
  *.png                         # Generated training plots
```

---

## 2. Docker Image Compilation

### 2.1 Build CPU image

Use this when `Dockerfile`, `requirements.txt`, or `requirements-dev.txt` changes:

```bash
docker compose build swarmball
```

For a clean rebuild without cache:

```bash
docker compose build --no-cache swarmball
```

### 2.2 Check that the container starts

```bash
docker compose run --rm swarmball
```

Expected output:

```text
Swarmball environment is ready
```

If Python code was changed only in `.py` files, rebuilding is usually not required because the project directory is mounted into the container as a volume.

---

## 3. Testing Protocols

### 3.1 Check Gymnasium / Stable-Baselines3 compatibility

```bash
docker compose run --rm swarmball python scripts/check_env.py
```

Expected output:

```text
Raw environment is compatible with Stable-Baselines3.
Flattened environment is compatible with Stable-Baselines3.
```

This verifies that:

- `reset()` returns correct values,
- `step()` returns correct values,
- `action_space` is valid,
- `observation_space` matches real observations.

### 3.2 Run automated tests

```bash
docker compose run --rm test
```

Expected result:

```text
passed
```

### 3.3 Run linter if configured

```bash
docker compose run --rm swarmball ruff check .
```

---

## 4. Code Verification Checklist

Before training or evaluation, verify:

1. `swarmball_env.py` uses the correct observation space.
2. `watch_model.py`, `evaluate_model.py`, and `evaluate_random.py` use the same environment parameters.
3. The model path points to `best_model`, not necessarily the final `model.zip`.
4. Random baseline and PPO model are evaluated with the same environment settings.
5. The evaluation seed is the same for random and PPO evaluation.

Recommended evaluation seed:

```bash
--seed 1000
```

Training seed can be different, for example:

```bash
--seed 42
```

---

## 5. Random Baseline Evaluation

The random baseline is an agent that selects random actions:

```python
action = env.action_space.sample()
```

It is used as a reference point to verify whether PPO learned a strategy better than random behavior.

Example:

```bash
docker compose run --rm swarmball python scripts/evaluate_random.py \
  --episodes 20 \
  --seed 1000 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0
```

Important metrics:

- mean reward,
- median reward,
- success count,
- enemy caught count,
- mean episode length.

---

## 6. PPO Training

Example training command:

```bash
docker compose run --rm swarmball python scripts/train_ppo.py \
  --timesteps 2000000 \
  --run-name ppo_2m_target_500_seed_42 \
  --lr 0.0003 \
  --gamma 0.995 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0 \
  --seed 42 \
  --eval-episodes 10
```

The training script saves outputs to:

```text
runs/ppo_2m_target_500_seed_42/
  model.zip
  best_model/
    best_model.zip
  checkpoints/
  eval_logs/
  monitor/
  tensorboard/
  config.json
```

Use `best_model.zip` for final evaluation and visualization because the final model is not always the best checkpoint.

---

## 7. Trained Model Evaluation

Evaluate the best PPO model:

```bash
docker compose run --rm swarmball python scripts/evaluate_model.py \
  --model-path runs/ppo_2m_target_500_seed_42/best_model/best_model \
  --episodes 20 \
  --seed 1000 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0
```

The evaluation should be compared with random baseline using the same parameters.

Example successful result:

```text
Random baseline:
Success count: 0/20

PPO best model:
Success count: 16/20
```

This indicates that PPO learned a strategy better than random behavior.

---

## 8. Best Checkpoint Analysis

Evaluation logs are stored in:

```text
runs/<run_name>/eval_logs/evaluations.npz
```

They can be used to find at which timestep the best checkpoint was reached.

Example result:

```text
Best evaluation:
timestep: 1200000
mean reward: 17379.30
```

This means that the best saved checkpoint was found around 1.2M timesteps.

---

## 9. Training Plot Generation

Generate training progress plots:

```bash
docker compose run --rm swarmball python scripts/plot_training_progress.py \
  --runs-dir runs \
  --output-dir plots \
  --window 10
```

The generated plot is saved to:

```text
plots/<run_name>_training_progress.png
```

If the plot is too noisy, increase the rolling window:

```bash
docker compose run --rm swarmball python scripts/plot_training_progress.py \
  --runs-dir runs \
  --output-dir plots \
  --window 20
```

The plot contains:

- raw episode reward,
- rolling mean reward,
- timesteps on the X axis,
- episode reward on the Y axis.

---

## 10. Graphical Display Mapping

The simulation visualization contains the following elements:

| Visual element | Meaning |
|---|---|
| Blue square | Goal object |
| Colored particles | Particle clusters controlled by the agent |
| Green vertical line | Finish line / `goal_target` |
| `FINISH` label | Target position |
| Enemy / wand | Object that catches the goal object from the left |
| Black terrain | Physical map / surface |

A successful episode occurs when the blue goal object reaches the green finish line before the enemy catches it.

---

## 11. Running Graphical Simulation with Trained Model

### 11.1 Linux with Docker

Give Docker access to the GUI:

```bash
xhost +local:docker
```

Run visualization:

```bash
docker compose run --rm simulation python scripts/watch_model.py \
  --model-path runs/ppo_2m_target_500_seed_42/best_model/best_model \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0 \
  --reset-on-done
```

Revoke access:

```bash
xhost -local:docker
```

### 11.2 Local Python

Activate virtual environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python scripts/watch_model.py \
  --model-path runs/ppo_2m_target_500_seed_42/best_model/best_model \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0 \
  --reset-on-done
```

### 11.3 Windows PowerShell

```powershell
.venv\Scripts\activate

python scripts/watch_model.py --model-path runs/ppo_2m_target_500_seed_42/best_model/best_model --goal-target 500 --enemy-acceleration 0.003 --enemy-max-speed 3.0 --reset-on-done
```

---

## 12. Expected Final Workflow

Recommended full verification sequence:

```bash
docker compose build swarmball

docker compose run --rm swarmball python scripts/check_env.py

docker compose run --rm test

docker compose run --rm swarmball python scripts/evaluate_random.py \
  --episodes 20 \
  --seed 1000 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0

docker compose run --rm swarmball python scripts/train_ppo.py \
  --timesteps 2000000 \
  --run-name ppo_2m_target_500_seed_42 \
  --lr 0.0003 \
  --gamma 0.995 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0 \
  --seed 42 \
  --eval-episodes 10

docker compose run --rm swarmball python scripts/evaluate_model.py \
  --model-path runs/ppo_2m_target_500_seed_42/best_model/best_model \
  --episodes 20 \
  --seed 1000 \
  --goal-target 500 \
  --enemy-acceleration 0.003 \
  --enemy-max-speed 3.0

docker compose run --rm swarmball python scripts/plot_training_progress.py \
  --runs-dir runs \
  --output-dir plots \
  --window 10
```

---

## 13. Notes

- Always evaluate random baseline and PPO model using the same environment parameters.
- Use `best_model.zip` for final results and demo.
- If the model fails due to observation shape mismatch, verify that the code version matches the model version.