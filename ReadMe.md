# Intro to Reinforcement Learning

A comprehensive, beginner-friendly journey through Reinforcement Learning algorithms - from basic concepts to state-of-the-art methods like PPO and SAC.

## What is this repository?

This repository provides an **intuitive and practical introduction to Reinforcement Learning** that bridges the gap between theory and implementation. Rather than overwhelming you with mathematical formulas, it focuses on building deep understanding through clear explanations, visual examples, and working code.

**Key Features:**
- 📚 **Theory-first approach** with intuitive explanations
- 💻 **Complete Python implementations** of all algorithms
- 🎮 **Tested on popular environments** (CartPole, LunarLander, Atari games)
- 📊 **Results and visualizations** showing algorithm performance
- 🔄 **Progressive learning path** from simple to advanced methods

## Who is this for?

- **Students** learning RL for the first time
- **Practitioners** who want to understand algorithms beyond just using libraries
- **Researchers** looking for clean, educational implementations
- **Anyone** who gets scared by dense mathematical papers but wants to truly understand RL

> *"I try to explain Reinforcement Learning to people who are interested in this field, but get scared by the massive amount of equations. I try to make the learning as intuitive as possible to create a deeper understanding of the topic."*

## Prerequisites

- **Python programming** (intermediate level)
- **Basic machine learning** concepts (neural networks, gradient descent)
- **Linear algebra and calculus** (high school level is sufficient)
- **Curiosity and patience** 😊

## Learning Path

The repository is structured as a 5-lesson progression, each building on the previous ones:

### 📖 [Lesson 1: Introduction & Black-Box Optimizers](Lesson-1/)
**What you'll learn:** RL fundamentals, essential components, and surprisingly effective simple methods

**Key Topics:**
- What is Reinforcement Learning?
- Agent, Environment, State, Action, Reward
- Cross-Entropy Method (CEM)
- Evolution Strategies (ES)
- Why these "simple" methods can be quite powerful

**Algorithms Implemented:**
- [`CEM`](Lesson-1/CEM/) - Learn through elite episode selection
- [`ES`](Lesson-1/ES/) - Evolution-inspired parameter optimization

**Environments:** CartPole, Acrobot, LunarLander

---

### 📖 [Lesson 2: Value-Based Methods](Lesson-2/)
**What you'll learn:** The mathematical foundations of RL and classical tabular methods

**Key Topics:**
- Markov Decision Processes (MDPs)
- Value functions V(s) and Q(s,a)
- Bellman equations and optimality
- Monte Carlo vs Temporal Difference learning
- The exploration vs exploitation dilemma

**Algorithms Implemented:**
- [`Monte Carlo`](Lesson-2/MonteCarlo/) - Learn from complete episodes
- [`SARSA`](Lesson-2/TD-Learning/) - On-policy temporal difference
- [`Q-Learning`](Lesson-2/TD-Learning/) - Off-policy temporal difference
- [`Expected SARSA`](Lesson-2/TD-Learning/) - Reduced variance updates
- [`Value Iteration`](Lesson-2/ValueIteration/) - Dynamic programming approach

**Environments:** CartPole, Taxi, Custom WindyGridWorld

---

### 📖 [Lesson 3: Deep Reinforcement Learning](Lesson-3/)
**What you'll learn:** Scaling RL to high-dimensional problems with neural networks

**Key Topics:**
- Why neural networks? The curse of dimensionality
- Deep Q-Networks (DQN) and its challenges
- Experience replay and target networks
- Advanced DQN variants and their motivations

**Algorithms Implemented:**
- [`DQN`](Lesson-3/DRL/1.%20DQN/) - The foundation of deep RL
- [`Double DQN`](Lesson-3/DRL/2.%20DDQN/) - Addressing overestimation bias
- [`Prioritized Experience Replay`](Lesson-3/DRL/3.%20PER/) - Learning from important experiences
- [`Dueling DQN`](Lesson-3/DRL/4.%20Dueling%20DQN/) - Separating state value from action advantages
- [`Multi-step DQN`](Lesson-3/DRL/5.%20Multi-step%20Return/) - Better credit assignment
- [`Distributional DQN`](Lesson-3/DRL/6.%20Distributional%20DQN/) - Learning return distributions
- [`Noisy Networks`](Lesson-3/DRL/7.%20Noisy%20Nets/) - Parameter space exploration
- [`RAINBOW`](Lesson-3/DRL/8.%20RAINBOW/) - Combining all improvements

**Environments:** CartPole, LunarLander, Acrobot, Atari games (Pong, Breakout, SpaceInvaders)

---

### 📖 [Lesson 4: Policy Gradient Methods](Lesson-4/)
**What you'll learn:** Direct policy optimization and the foundation of modern RL

**Key Topics:**
- Why learn policies directly?
- Policy gradient theorem and its derivation
- Variance reduction techniques
- Actor-Critic architectures
- The importance of parallel environments

**Algorithms Implemented:**
- [`REINFORCE`](Lesson-4/SPG/REINFORCE/) - The foundation of policy gradients
- [`Advantage Actor-Critic (A2C)`](Lesson-4/SPG/AAC/) - Reducing variance with baselines
- [`Asynchronous Advantage Actor-Critic (A3C)`](Lesson-4/SPG/A2C/) - Parallel training
- [`Proximal Policy Optimization (PPO)`](Lesson-4/SPG/PPO/) - Safe policy updates

**Environments:** CartPole, LunarLander, Acrobot, Atari games

---

### 📖 [Lesson 5: Advanced Policy Methods](Lesson-5/)
**What you'll learn:** State-of-the-art algorithms for continuous control

**Key Topics:**
- Continuous action spaces and their challenges
- Deterministic vs stochastic policies
- Modern actor-critic methods
- Entropy regularization and soft policies

**Theory Covered:**
- Deterministic Policy Gradient (DPG)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)
- Discrete SAC

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/fenixkz/rl_journey.git
   cd rl_journey
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start with Lesson 1**
   ```bash
   cd Lesson-1
   # Read theory.md first, then explore the implementations
   ```

4. **Run an algorithm**
   ```bash
   cd CEM
   python train.py
   ```

## Repository Structure

```
├── Lesson-1/           # Introduction & Black-box methods
│   ├── theory.md       # Detailed theoretical explanations
│   ├── CEM/           # Cross-Entropy Method implementation
│   ├── ES/            # Evolution Strategies implementation
│   └── figures/       # Visualizations and diagrams
│
├── Lesson-2/           # Value-based methods
│   ├── theory.md
│   ├── MonteCarlo/    # Monte Carlo methods
│   ├── TD-Learning/   # SARSA, Q-learning, Expected SARSA
│   ├── ValueIteration/ # Dynamic programming
│   └── figures/
│
├── Lesson-3/           # Deep Reinforcement Learning
│   ├── theory.md
│   └── DRL/           # All DQN variants + RAINBOW
│       ├── 1. DQN/
│       ├── 2. DDQN/
│       ├── 3. PER/
│       └── ... (8 total algorithms)
│
├── Lesson-4/           # Policy Gradient Methods
│   ├── theory.md
│   └── SPG/           # Stochastic Policy Gradient methods
│       ├── REINFORCE/
│       ├── AAC/       # Advantage Actor-Critic
│       ├── A2C/       # Asynchronous AC
│       └── PPO/       # Proximal Policy Optimization
│
└── Lesson-5/           # Advanced Policy Methods
    └── theory.md       # DPG, DDPG, TD3, SAC theory
```

## What Makes This Different?

1. **Intuition First:** Every algorithm is explained with intuition before diving into math
2. **Complete Implementations:** All code is written from scratch for educational clarity
3. **Progression Matters:** Each lesson builds naturally on previous concepts
4. **Real Results:** Every algorithm includes training results and performance plots
5. **Modern Relevance:** Covers algorithms actually used in practice today

## 📊 Algorithms & Performance

| Algorithm | Type | Best For | Sample Efficiency | Stability |
|-----------|------|----------|------------------|-----------|
| CEM | Black-box | Simple problems | ⭐⭐ | ⭐⭐⭐ |
| Q-Learning | Value-based | Discrete actions | ⭐⭐⭐ | ⭐⭐ |
| DQN | Value-based | Discrete + Images | ⭐⭐⭐ | ⭐⭐⭐ |
| RAINBOW | Value-based | Discrete + Complex | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PPO | Policy-based | General purpose | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| SAC | Policy-based | Continuous control | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Tested Environments

- **Classic Control:** CartPole, Acrobot, LunarLander
- **Discrete Environments:** Taxi, Custom GridWorlds
- **Atari Games:** Pong, Breakout, SpaceInvaders
- **Continuous Control:** (Theory covered, implementations welcome!)

## Contributing

This is an educational resource - contributions are welcome!

- **Found a bug?** Open an issue
- **Improved explanation?** Submit a pull request
- **New algorithm?** Follow the existing structure
- **Better visualizations?** Always appreciated!

## Recommended Reading Order

1. Start with [`Lesson-1/theory.md`](Lesson-1/theory.md) - even if you know RL basics
2. Run at least one algorithm from each lesson
3. Compare results across different algorithms
4. Experiment with hyperparameters
5. Try implementing variants or improvements

## Key Learning Outcomes

After completing all lessons, you will:

- ✅ Understand the fundamental principles of RL
- ✅ Know when to use different types of algorithms
- ✅ Be able to implement RL algorithms from scratch
- ✅ Understand the trade-offs between exploration and exploitation
- ✅ Grasp the challenges of scaling RL to complex problems
- ✅ Appreciate why modern algorithms like PPO and SAC work so well

## License

MIT License - feel free to use this for learning, teaching, or research!

## Acknowledgments

- Implementations tested on OpenAI Gym environments
- Inspired by classic RL textbooks but focused on practical understanding
- Thanks to the RL community for developing these amazing algorithms

---

**Ready to start your RL journey? Begin with [Lesson 1](Lesson-1/theory.md)!** 🚀
