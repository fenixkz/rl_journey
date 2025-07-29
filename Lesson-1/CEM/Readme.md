# Cross-Entropy Method (CEM) for Reinforcement Learning

## Table of Contents
1. [Introduction to Gym](#introduction-to-gym)
2. [CartPole-v1 Environment](#cartpole-v1-environment)
3. [Cross-Entropy Method (CEM)](#cross-entropy-method-cem)
4. [Code Structure](#code-structure)
5. [How to Run](#how-to-run)
6. [Understanding the Results](#understanding-the-results)


## Introduction to Gym

**Gymnasium** (formerly OpenAI Gym) is a Python library that gives you a playground of environments for testing reinforcement learning algorithms. Think of it as a collection of games and simulations where you can train your AI to get better at different tasks.

### Key Concepts:

- **Environment**: The world your agent interacts with (like a game or a robot simulation)
- **Observation Space**: What your agent can "see" about the world (the state)
- **Action Space**: What your agent can "do" (the possible moves)
- **Step Function**: How the environment reacts when your agent takes an action
- **Reset Function**: How to start a new game or episode

### Basic Gym Usage:
```python
import gymnasium as gym

# Create the environment
env = gym.make('CartPole-v1')

# Start a new episode and get the initial state
state = env.reset()

# Pick an action (here, just a random one)
action = env.action_space.sample()
# Take the action and see what happens
next_state, reward, done, info = env.step(action)
```

## CartPole-v1 Environment

CartPole is a classic challenge in reinforcement learning, and it's a great place to start. If your algorithm can't solve CartPole, it's probably not working right!

### The Problem
Picture a cart that can move left and right, with a pole balanced on top. Your job is to keep the pole upright by pushing the cart left or right at each step.

### Environment Details:

**Observation Space (State):**
- Cart Position: How far the cart is from the center
- Cart Velocity: How fast the cart is moving
- Pole Angle: How tilted the pole is
- Pole Angular Velocity: How quickly the pole is rotating

So, the state is just an array of 4 numbers, like `[ 0.02972086,  0.01830195, -0.02571287, -0.0120202 ]`.

**Action Space:**

- Action 0: Push the cart to the left
- Action 1: Push the cart to the right

**Rewards:**
- You get +1 for every timestep the pole stays up. The longer you balance it, the higher your score!
- The episode ends if:
  - The pole falls more than 15 degrees
  - The cart moves more than 2.4 units from the center
  - You reach 500 steps (so you can't play forever)
  - The cart goes outside the display area

**Success Criteria:**
- Average reward of 495+ over 100 episodes means you've really nailed it
- In our code, we consider it solved if the mean over 100 episodes is above 400 for 2 steps in a row


## Code Structure

### [`agent.py`](agent.py)
This file has the `Agent` class, which uses a neural network policy to learn how to act.

**Key Components:**
- **3-layer neural network**: Takes in the state, processes it through hidden layers, and outputs action probabilities
- **`choose_action()` method**: Picks what to do next, given the current state
- **`learn()` method**: Updates the network using the best episodes

### [`train.py`](train.py)
This is where the main training loop lives, using the CEM algorithm.

**Key Functions:**
- **`play_one_episode()`**: Runs through one episode and records (state, action, reward) at each step
- **`select_elites()`**: Picks out the top-performing episodes
- **Training loop**: Generates episodes, selects elites, and trains the policy
- **Evaluation**: Tests how well the trained agent performs

## How to Run

### Prerequisites:
```bash
pip install torch gymnasium numpy matplotlib tqdm
```

### Running the Code:
```bash
python train.py
```

### What You'll See:
- A progress bar showing training steps and average rewards
- Training will stop when the mean reward goes above 200 for 5 steps in a row
- At the end, you'll see how well the agent does over 10 episodes
- You'll also get a plot showing how learning progressed

## Understanding the Results

### Training Progress:
- **Mean Reward**: The average score across all episodes in the current round
- **Target**: Aim for 400+ consistently (CartPole-v1 is officially "solved" at 495+)
- **Elite Percentile**: We keep the top 30% of episodes (the 70th percentile and above)

### What Good Results Look Like:
- The mean reward should climb as training goes on
- You should hit 200+ within 50-100 training steps
- In the final evaluation, the agent should consistently get rewards above 200

### Typical Learning Curve:
```
Step 0-10:   Low rewards (50-100) - The agent is just exploring randomly
Step 10-20:  Rapid improvement (100-400) - It's starting to figure things out
Step 20+:    Convergence (400+) - Mastering the task!
```

## Key Takeaways

1. **RL is about learning by doing** - The agent gets better by trying things and seeing what works
2. **CEM is simple but powerful** - "Keep doing what works" is a surprisingly effective strategy
3. **CartPole teaches the basics** - It's all about balance, control, and making decisions step by step
4. **Gym makes RL easy to experiment with** - A standard interface lets you try out different algorithms quickly

This project shows how even a straightforward algorithm can solve a classic challenge, and gives you a solid starting point for exploring more advanced reinforcement learning methods!

