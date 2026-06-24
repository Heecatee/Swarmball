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


## Docker development setup

This project contains a Docker-based development setup to make it easier to run the project on different machines.

The default Docker image is CPU-based and is recommended for normal development, testing and linting. An optional GPU image can be used later for model training with CUDA.

### Build the Docker image

```bash
docker compose build
```

Run this after cloning the repository for the first time.
You should rebuild the image after changing:

```text
Dockerfile
Dockerfile.gpu
requirements.txt
requirements-dev.txt
docker-compose.yml
```
You do not need to rebuild the image after normal Python code changes, because the project directory is mounted into the container.

### Verify the environment
```bash
docker compose run --rm swarmball
```
Expected output:
```text
Swarmball environment is ready
```
This checks that the basic Python environment and project dependencies are available inside the container.

### Run tests
```bash
docker compose run --rm test
```
### Run linting
```bash
docker compose run --rm lint
```
Note: the legacy codebase currently contains existing Ruff warnings/errors. The lint command is available as a development tool, but fixing all lint issues is a separate cleanup task.

### Run the simulation
```bash
docker compose run --rm simulation
```
The simulation uses PyGame. On Linux/X11, graphical output from Docker requires access to the host display:
```bash
xhost +local:docker
docker compose run --rm simulation
xhost -local:docker
```
The simulation service uses X11 display forwarding, dummy audio driver and software rendering.

Note: the current simulation may still fail with a known legacy Pymunk compatibility error:
```text
Exception: Unsupported type <class 'list'>
```

### Running the simulation on Windows
#### Option 1: Run Docker through WSL2

Recommended setup:

```text
Windows 11
WSL2 Ubuntu
Docker Desktop
Docker Desktop WSL integration enabled
```

Open the project inside WSL2 Ubuntu and run:

```bash
docker compose build
docker compose run --rm swarmball
docker compose run --rm test
```

To try running the PyGame simulation from Docker:
```bash
docker compose run --rm simulation
```

On Windows 11 with WSLg, graphical applications from WSL usually work automatically. However, GUI support from Docker may still depend on the local Docker Desktop and WSL configuration.

If the simulation does not open a window from Docker, use Option 2 and run the simulation locally.

#### Option 2: Run the graphical simulation locally

Docker is still recommended for environment checks, tests and linting:
```bash
docker compose run --rm swarmball
docker compose run --rm test
docker compose run --rm lint
```

But the PyGame simulation can be run directly on Windows with a local Python environment.

Create and activate a virtual environment:

```PowerShell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch is not listed in requirements.txt, install CPU PyTorch separately:
```PowerShell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Run the simulation:
```PowerShell
python -m environment.simulation.simulation
```

Note: the simulation may still fail with a known legacy Pymunk compatibility issue.

This means the environment starts correctly, but the old simulation code still needs to be modernized for the current Pymunk API.

### GPU training image

The default Docker image uses CPU PyTorch for portability.

For GPU training, use the optional GPU image:
```bash
docker compose build gpu
docker compose run --rm gpu
```

Before using the GPU image, the host machine must have:
```text
NVIDIA GPU
NVIDIA driver
NVIDIA Container Toolkit
Docker GPU support
```

Check the host GPU:
```bash
nvidia-smi
```

If Docker can access the GPU, the GPU service should print:
```text
CUDA available: True
```

Later, training scripts can be run with:
```bash
docker compose run --rm gpu python3 scripts/train_ppo.py
```

or:
```bash
docker compose run --rm gpu python3 scripts/train_sb3.py
```

### Useful commands
```bash
docker compose build
docker compose run --rm swarmball
docker compose run --rm test
docker compose run --rm lint
docker compose run --rm simulation
docker compose run --rm gpu
```

To open a shell inside the container:
```bash
docker compose run --rm swarmball bash
```
