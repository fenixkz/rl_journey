# Value and Policy Iteration
## Theory

First off, let's define the playground for many Reinforcement Learning algorithms. RL often focuses on problems that can be modeled as Markov Decision Processes (MDPs). This name comes from a key assumption they satisfy: the Markov Property.

The Markov Property is crucial and can be summed up like this: the future is independent of the past, given the present.

Mathematically, it means the probability of transitioning to the next state $s_{t+1}$ and receiving reward $r_{t+1}$ depends only on the current state $s_t$ and the action taken $a_t$, not on the entire history of states and actions before that.

$P(s_{t+1}, r_{t+1} | s_t, a_t) = P(s_{t+1}, r_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)$

In simple words: all the useful information from the history needed to predict the immediate future is captured within the current state $s_t$. Knowing $s_t$ and the action $a_t$ makes the older history redundant for predicting what happens next. Problems satisfying this property are what we call Markovian, and they form the basis for algorithms like Value and Policy Iteration.

Let's review two examples: a chess game and weather forecasting.

### Chess Game

In chess, the current configuration of all pieces on the board (along with minor flags for things like castling rights or en passant possibilities) serves as the state $s_t$. Knowing this complete state is enough to determine all legal moves and the potential outcomes of those moves according to the rules. How the pieces arrived at their current positions (e.g., why a piece was taken earlier) doesn't change the rules or possibilities moving forward from the current board state. Given the state $s_t$, our policy $\pi_\theta$ can choose an action $a_t$ (a move), which then, according to the game rules (the transition dynamics), leads us to a new state $s_{t+1}$. Chess, under this definition, is essentially Markovian.

### Weather Forecast

Weather forecasting is interesting because its "Markov-ness" really depends on how we define the state. Because, meteorologists don't just look at today's weather; they examine historical data, pressure systems, wind patterns, etc.

#### Non-Markovian State 

If we define our state $s_t$ simply as $S \in \{RAIN, SUNNY, CLOUDY\}$ for today, predicting tomorrow becomes very hard. Knowing it rained yesterday too might significantly change our prediction for tomorrow compared to if yesterday was sunny. The history clearly matters, so this simple state definition isn't Markovian.

#### Towards a Markovian State 

To make it Markovian, we'd need to enrich the state to include all relevant information. Perhaps $S = (\text{Temp}_t, \text{Pressure}_t, \text{Wind}_t, \text{Humidity}_t, \text{RecentPrecip}_{t-k}, \dots)$. The idea is to include enough features in the current state definition $s_t$ so that the previous states $s_{t-1}, s_{t-2}, \dots$ offer no additional predictive power. Can we ever capture everything perfectly? Maybe not, which is why modelling complex real-world systems often involves assumptions and approximations to make them "close enough" to Markovian for our algorithms to work well.


## Simple Environment Example & Policy Types

Imagine a game with three states: $s_0, s_1, s_2$. You start in $s_0$, and from this state you have two actions: $a_0, a_1$. Action $a_0$ transitions you to $s_1$ with reward $R_1$. From state $s_1$ you have only one option: go back to $s_0$ with reward $0$. Action $a_1$ transitions you to $s_2$ with reward $R_2$. From $s_2$ you also have only one option: terminate the game [let's assume reward on this transition is 0, the $R_2$ was received upon entering $s_2$]. Assume $R_2 > R_1 / (1 - \gamma^2)$ [don't worry about gamma, it is just a condition that makes the second action $a_1$ strictly better].

What's the optimal policy here?

Our human intuition might spot the loop $s_0 \xrightarrow{a_0, R_1} s_1 \xrightarrow{a_0, 0} s_0$. We think, "Maybe I can collect many $R_1$ rewards by looping, and then, when I'm 'tired', take action $a_1$ from $s_0$ to get $R_2$ and finish."

But can a standard RL algorithm find this specific strategy? 

Policies in Standard MDPs are (typically) deterministic and Markovian: A standard policy $\pi(a|s)$ bases its decision only on the current state $s$. If the environment itself is Markovian (like the one described), then the optimal policy is also guaranteed to be Markovian. It doesn't need history.
Your "Human" Strategy Isn't Markovian (for state $s_0$): Your strategy "loop for a while, then switch" depends on how long you've been looping (history), not just on being in state $s_0$.
What the RL Algorithm Finds: Given state $s_0$, a standard policy $\pi(a|s_0)$ must choose either always a_0 or always a_1. It compares the total expected discounted value of always looping ($V = R_1 / (1-\gamma^2)$) versus the value of terminating (V = R_2). Since we assumed $R_2 > R_1/(1-\gamma^2)$, the optimal Markovian policy is to always choose $a_1$ from state $s_0$. It cannot implement the "loop then switch" plan.

Why? The policy function $\pi(a|s)$ doesn't have access to the history or the total accumulated reward unless that information is part of the state $s$. It makes the best decision based only on the information contained in $s$.

Can RL Ever "Act Like Us"?

Yes! But it requires acknowledging that the "loop then switch" strategy implies the simple state $s \in \{s_0, s_1, s_2\}$ isn't sufficient (it's not Markovian for that strategy). To achieve this behavior, we would need to:

Enhance the State Representation: Define the state to include history, for example, $s' = (s, \text{loop\_count})$. Now the policy $\pi(a|s')$ can learn to choose a_0 when loop_count is low and a_1 when loop_count is high, because the necessary information is in the state.

Use stochastic policy: a policy does not have to be deterministic. A stochastic policy can choose actions with some probability given the current state. For example, in this case, given state $s_0$, the policy can coverge to $\pi(a_0|s_0) = 0.9$ and $\pi(a_1|s_0) = 0.1$, meaning that with 90% chance the agent will choose $a_0$. This way it will earn more rewards, until eventually it randomly (with p = 0.1) chooses to go to $s_2$ and terminate the game.

## State values and state-action values

In order to start understanding how to solve the MDP environments, we need to introduce some new concepts. First one is $V(s)$ or state value. This concept can be intuitively understood by the usefullness of the state. The higher $V(s)$ the higher attractivenes of that state. 

$$
V(s) = \mathbb{E}[\sum_{t=0}^\infty R_t | S_t = s]
$$

Mathematically, we say that state value is expectation (basically a mean for random variables) of total cumulative reward if we start from state $s$ and then act according to our policy. 

Mathematically, this is often written as:

$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid S_t = s \right]
$

where $\gamma$ is the discount factor ($0 \leq \gamma < 1$), $r_{t+k+1}$ is the reward received $k$ steps into the future, and the expectation is taken over the stochasticity of the environment and the policy $\pi$.

### State-Action Value Function

Closely related is the state-action value function, $Q(s, a)$, which represents the expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$:

$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid S_t = s, A_t = a \right]
$

The $Q$-function tells us how good it is to take a particular action in a particular state, while the $V$-function tells us how good it is to be in a particular state (assuming we act according to policy $\pi$).

### Bellman Equations

The value functions satisfy important recursive relationships known as the Bellman equations. In other words, if our $V(s_t)$ depends on $V(s_{t+1})$, then there is recursion, right?

$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid S_t = s \right] \\
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=1}^\infty r_{t+1} + \gamma^k r_{t+k+1} \mid S_t = s \right] \\
V^\pi(s) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid S_t = s \right]
$

So, our $V(s_t)$ is a reward that we get when we transit to some next state $s_{t+1}$ plus the $V(s_{t+1})$ discounted by $\gamma$. Since, there could be several possible next states and there are various actions that can lead to those states, we should find an expectation over all possible actions and next states.

For a given policy $\pi$, the Bellman equation for the state value function is:

$
V^\pi(s) = \sum_{a} \left\{ \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right] \right\}
$

This equation expresses the value of a state as the expected immediate reward plus the discounted value of the next state, averaged over all possible actions and transitions according to the policy and environment dynamics.

Similarly, the Bellman equation for the state-action value function is:

$
Q^\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]
$

A quick example: the agent is in state $`s`$, it has two actions to choose from $a_1, a_2$. Let's say that if agent chooses $a_1$ it flips a coin and based on result it can either be in $s_1$ and getting reward = 10 or $s_2$ and getting reward = -4, but taking action $a_2$ always leads to state $s_3$ with reward = 2. And finally, let's say that there is already some defined policy that favours second action $a_2$ giving to it 70% probability. 

So, our $Q(s, a)$ would be (assuming V values for other states are zeros, they are useless in other words):

$$
Q(s, a_1) = 0.5 \cdot (10 + V(s_1)) + 0.5 \cdot (-4 + V(s_2)) = 3  \\
Q(s, a_2) = 1 \cdot (2 + V(s_3)) = 2
$$

And finally, our $V(s)$ is simply a sum of both Q(s, a) weighted by their probabilities:

$$
V(s) = \sum_{a} \pi(a|s) Q(s, a) = 0.3 \cdot 3 + 0.7 \cdot 2 = 1.7
$$

So, under this defined policy our values for that state is 1.7. Question, is it optimal (the best) policy?

#### How to derive $Q$ from $V$ and $V$ from $Q$

The two value functions are tightly connected. In fact, if you know one, you can compute the other—at least in principle.

- **Deriving $Q$ from $V$:**  
  If you know $V^\pi(s)$ for all $s$, you can compute $Q^\pi(s, a)$ using the Bellman equation:
  $$
  Q^\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]
  $$
  That is, for each possible next state $s'$ and reward $r$, you sum over the probability of transitioning to $s'$ and receiving $r$, and add the discounted value of $s'$.

- **Deriving $V$ from $Q$:**  
  If you know $Q^\pi(s, a)$ for all $(s, a)$, you can compute $V^\pi(s)$ by averaging over the policy:
  $$
  V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)
  $$
  That is, you take the expected $Q$ value under the current policy's action probabilities.

#### Why is this sometimes hard in practice?

In small, tabular environments, these relationships are straightforward to use. But in real-world or large-scale problems, several challenges arise:

- **Unknown Transition Probabilities:**  
  The formula for $Q^\pi(s, a)$ from $V^\pi(s)$ requires knowing $P(s', r | s, a)$ for all possible next states and rewards. In most practical RL problems, the environment is a black box, so these probabilities are not available.

- **Large or Continuous State/Action Spaces:**  
  If the state or action space is very large (or continuous), you cannot store $V$ or $Q$ in a table. Computing the sums or expectations exactly becomes infeasible, and you must use function approximation (like neural networks), which introduces estimation errors.

## Example

Okay, time to see some practical example to solidify these formulas. The idea is quite simple, we just defined two new concepts one is saying how good it is to be in state $s$ and the second one is saying how good it is to take action $a$ from state $s$. All these expectations are a way to work with not deterministic problems, but with stochastic. 

Let's consider a simple stochastic environment with two states: $S_1$ and $S_2$.

### Environment Description

- **States:** $S_1$, $S_2$
- **Actions:** $a_1, a_2$ (available in both states)
- **Transitions and Rewards:**
  - From $S_1$, taking action $a_1$:
    - With probability $0.7$, move to $S_1$ and get reward $-7$
    - With probability $0.3$, move to $S_2$ and get reward $20$
  - From $S_1$, taking action $a_2$:
    - With probability $1.0$, move to $S_2$ and get reward $1$
  - From $S_2$, both actions:
    - With probability $1.0$, stay in $S_2$ and get reward $0$
- **Discount factor:** $\gamma = 0.9$
- **Policy:** Let's consider two policies:
  - $\pi_1$: Always take $a_1$ in $S_1$
  - $\pi_2$: Always take $a_2$ in $S_1$

### Step 1: Write the Bellman Equations

Let $V^{\pi}(S_1)$ and $V^{\pi}(S_2)$ be the value functions for each state under policy $\pi$.

#### For $S_2$ (any policy):
- Only one possible transition:

  $
  V^{\pi}(S_2) = \mathbb{E}[r + \gamma V^{\pi}(S_2)] = 0 + 0.9 \cdot V^{\pi}(S_2)
  $

  $
  V^{\pi}(S_2) = 0.9 \cdot V^{\pi}(S_2)
  $

  $
  V^{\pi}(S_2) = 0
  $

#### For $S_1$:

##### Policy $\pi_1$ (always $a_1$):
- Two possible transitions:

  $
  V^{\pi_1}(S_1) = 0.7 \cdot [-7 + 0.9 V^{\pi_1}(S_1)] + 0.3 \cdot [20 + 0.9 V^{\pi_1}(S_2)] + 0 \cdot 1 \cdot [1 + 0.9 V^{\pi_1}(S_2)]
  $

  Substitute $V^{\pi_1}(S_2) = 0$:

  $
  V^{\pi_1}(S_1) = 0.7 \cdot (-7 + 0.9 V^{\pi_1}(S_1)) + 0.3 \cdot (20 + 0)
  $

  $
  V^{\pi_1}(S_1) = 0.7 \cdot -7 + 0.7 \cdot 0.9 V^{\pi_1}(S_1) + 0.3 \cdot 20
  $

  $
  V^{\pi_1}(S_1) = -4.9 + 0.63 V^{\pi_1}(S_1) + 6
  $

  $
  V^{\pi_1}(S_1) = 1.1 + 0.63 V^{\pi_1}(S_1)
  $

  $
  V^{\pi_1}(S_1) - 0.63 V^{\pi_1}(S_1) = 1.1
  $

  $
  0.37 V^{\pi_1}(S_1) = 1.1
  $

  $
  V^{\pi_1}(S_1) = \frac{1.1}{0.37} \approx 2.97
  $

##### Policy $\pi_2$ (always $a_2$):
- Only one possible transition:

  $
  V^{\pi_2}(S_1) = 0 \cdot (0.7 \cdot [-7 + 0.9 V^{\pi_2}(S_1)] + 0.3 \cdot [20 + 0.9 V^{\pi_2}(S_2)]) +  1.0 \cdot [1 + 0.9 V^{\pi_2}(S_2)]
  $

  Substitute $V^{\pi_2}(S_2) = 0$:

  $
  V^{\pi_2}(S_1) = 1 + 0.9 \cdot 0 = 1
  $

### Step 2: Compute $Q$-values

Let's compute $Q(S_1, a_1)$ and $Q(S_1, a_2)$ using the Bellman equation for both policies:

$
Q^{\pi}(S_1, a_1) = \pi^{x}(a_1 | s_1) \cdot (0.7 \cdot [-7 + 0.9 V^{\pi}(S_1)] + 0.3 \cdot [20 + 0.9 V^{\pi}(S_2)])
$

$
Q^{\pi}(S_1, a_2) = \pi^{x}(a_2 | s_1) \cdot [1 + 0.9 V^{\pi}(S_2)]
$

Using $V(S_2) = 0$ for both policies:

- For $a_1$:

  $
  Q^{\pi_1}(S_1, a_1) = 0.7 \cdot [-7 + 0.9 \times 2.97] + 0.3 \cdot 20 \approx 10.4 \\ 
  Q^{\pi_2}(S_1, a_1) = 0 \cdot (1 \cdot [1 + 0.9 \times 0]) = 0
  $

- For $a_2$:

  $
  Q^{\pi_1}(S_1, a_2) = 0 \cdot [1 + 0.9 \times 0] = 0 \\
  Q^{\pi_2}(S_1, a_2) = 1 \cdot [1 + 0.9 \times 0] = 1
  $


### Optimal Value Functions

Once, we understood what is $V(s)$ and $Q(s,a)$ we can understand the optimal value functions. Again, the goal in reinforcement learning is typically to find an optimal policy $\pi^*$ that maximizes the expected return from every state. The corresponding optimal value functions are:

- **Optimal state value function:** $V^*(s) = \max_\pi V^\pi(s)$
- **Optimal state-action value function:** $Q^*(s, a) = \max_\pi Q^\pi(s, a)$

The Bellman optimality equations for these are:

$
V^*(s) = \max_a \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^*(s') \right] = \max_a Q^*(s, a)
$

$
Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right] = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^*(s') \right]
$

The interpretation is quite simple and obvious, based on our Q-values for each action we just choose the action with the highest Q-value. 

So, in our quick example above, the defined policy was not the optimal, the optimal policy would be the one that chooses action based on the highest Q-value. 


## Value Iteration

Okay, now the we know value and state-values as well as their optimal definitions, we can make a simple, yet effective algorithm for finding the optimal values for each state. Then, based on the optimal values we can easily deduce Q-values and based on Q-values we can always choose the best action.

And the pseudo-algorithm is quite simple:

- Initialize the initial values for each state as zero: 
$
\forall s \in S, \quad V(s) = 0
$

- Repeat:
    - For each state $s \in S$:
        - $` Q^*(s, a) \leftarrow \sum_{s', r} P(s', r | s, a) \left (r + \gamma * V^*(s') \right)`$
        - $V^*(s) \leftarrow \max_a Q^*(s, a)$

And that magically converges to the optimal values for each state! After you know $V^*(S)$ you compute Q-values and the take argmax to choose an action.
