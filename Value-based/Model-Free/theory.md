# Model-Free Learning and TD 

## Model-Free learning

First of all, let's recall what we learned so far. We learned a simple, yet powerful algorithm called Value Iteration. This algorithm is based on the Bellman Optimality Equations, which define the optimal state-value function $V^*(s)$ and optimal action-value function $Q^*(s,a)$.

### Value Iteration

Value Iteration assumes a perfect model of the MDP is available (knowing all transition probabilities $P(s', r | s, a)$ and rewards $r$). It uses a fixed-point algorithm to find the optimal state-value function $V^*(s)$. The Bellman Optimality Equations define this optimal value:

$Q^*(s,a) = \sum_{s', r} P(s', r | s, a) \left (r + \gamma V^*(s') \right)$  (Optimal value of taking action 'a' in state 's', then acting optimally)

$V^*(s) = \max_a Q^*(s, a)$  (Optimal value of state 's' is the value of the best action from 's')

Value Iteration finds $V^*$ by repeatedly applying the Bellman update across all states:

$V_{k+1}(s) \leftarrow \max_a \sum_{s', r} P(s', r | s, a) \left( r + \gamma V_k(s') \right)$

The major limitation of this method is its reliance on the model term: $\sum_{s', r} P(s', r | s, a)$. In the majority of interesting problems, the environment's internal probability distributions for transitions and rewards are unknown. We typically just act based on observations and see what happens next (<s, a, r, s'>). Without knowing the model (P and R), Value Iteration cannot be directly applied. Using VI with an inaccurate learned model would yield results optimal only for that inaccurate model, not necessarily for the true environment.

Is there a way to overcome this issue and learn optimal behavior without explicitly knowing the environment's dynamics? Yes, this is where model-free methods come in.

### Monte Carlo Method

How can we learn without knowing the model? One major approach involves Monte Carlo (MC) methods.

So, we remember that state-value is the total return $G$ (cumulative reward) from a given state onwards. Then what stops us from playing one full episode (start to end), record all data $<s, a, r, s'>$, and sum all rewards to get $G$. Then, so the first state has value = total $G$, second state in our trajectory would be $G$ minus reward that we got from moving from previous state to this state and so on. This way we get an estimate for each value, by playing many other episodes we just average all state-values. 

#### Core Idea 
MC methods learn from complete episodes. An agent follows a policy $\pi$ to generate a full trajectory $<s_0, a_0, r_1, s_1, ..., s_{T-1}, a_{T-1}, r_T>$. Only once the episode is finished (at terminal step T), learning can occur for that episode.

To estimate $V^\pi(s)$, MC methods calculate the actual total discounted return $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$ starting from each time step t where state $s$ was visited. The value $V^\pi(s)$ is then estimated simply by averaging these observed returns $G_t$ over many episodes. Similarly, $Q^\pi(s, a)$ can be estimated by averaging returns following visits to the specific state-action pair $(s, a)$. Question may arise: how to deal with cases when the state was visited several times during the episode? The answer may be simple, but effective: we just either average all returns from the same state or update only the first occurence and ignore later.

MC updates use only actual, complete returns (G_t) from experience. They don't use estimates of other states' values to update the current state's value. That makes this method unbiased, but it has a high variance. High variance is a result of noise in the individual trajectories: each episode might take a different path due to stochasticity in the environment or policy, leading to different returns even from the same state. Furthermore, random events far in the future affect the estimate.

This reliance on complete episodes explains why MC methods might feel different from a simple notion of "play and learn based on your reward immediately". Updates only happen retrospectively after the final outcome is known. So, in other words we are not learning while we are playing, we only learn once we finished. 

### Temporal difference (TD) learning 

So, Monte Carlo methods require waiting until the end of an episode to update value estimates. The obvious question is: can we learn during the episode, essentially learning as we play? The answer is yes, using Temporal Difference (TD) learning.

The core idea is to use the Bellman equation structure, but instead of needing the full model, we update our estimates based on other estimates – a process called bootstrapping. This might sound similar to Value Iteration, where we updated $V_k$ using $V_{k-1}$. However, Value Iteration required the environment model (P and R) to compute the expected values across all possible next states and rewards.

In model-free TD settings, we don't have the model. Instead, we learn directly from the experience tuples $<s, a, r, s'>$ as we generate them. After taking action $a$ in state $s$ and observing the immediate reward $r$ and the next state $s'$, we can immediately use this information to improve our estimate for the value of state $s$ (or state-action pair $(s,a)$). We learn one step at a time, using the observed transition rather than a known probability distribution.

#### Estimating V(s) vs. Q(s,a) in Model-Free Learning

What should we estimate, $V(s)$ or $Q(s,a)$? While learning $V^*(s)$ (the optimal state value) might seem sufficient, consider how an agent chooses an action. If it only knows $V^*(s)$, to decide the best action $a$ from state $s$, it still needs a model to look ahead one step:

$\pi^*(s) = \arg\max_a \sum_{s', r} P(s', r | s, a) \left( r + \gamma V^*(s') \right)$

Without the model ($P$ and $R$), knowing $V^*(s)$ alone isn't enough to determine the best action.

However, if we directly estimate the optimal action-value function $Q^*(s,a)$, selecting the best action becomes simple and model-free:

$\pi^*(s) = \arg\max_a Q^*(s, a)$

The agent just needs to compare the learned Q-values for all actions possible in the current state s and pick the best one. Therefore, for model-free control (finding the optimal policy), learning $Q$-values is generally much more direct and useful than learning $V$-values.

#### Bellman Optimality Equation for Q-values:

The optimal Q-function must satisfy its own Bellman Optimality Equation:

$Q^*(s,a) = \sum_{s', r} P(s', r | s, a) \left( r + \gamma \max_{a'} Q^*(s', a') \right)$

(Note: This relates $Q^*(s,a)$ to the maximum Q-value possible from the next state $s'$, because $V^*(s') = \max_{a'} Q^*(s', a')$).

So, this part $\sum_{s', r} P(s', r | s, a)$ is basically a weighted average of all possible next state-action values. The weights are the probabilities of transitioning to each next state-action pair.
Since in our tuple there is only one next state, we bypass this averaging and work only with one next state. But this is not the optimal $Q*$, it is just our estimate based on one sample.

$\text{TD Target} = ( r + \gamma \max_{a'} Q^*(s', a'))$

The TD Target is a noisy, sample-based estimate of what the true expected value $Q^*(s,a)$ should be. Because the environment can be stochastic (different r or s' could occur from the same s,a) and our current Q estimates might be wrong, the TD Target calculated from one step will fluctuate.

So, for stability of learning we need a technique to smooth out this noise, one of the simplest and widely used technique is moving average:

$$
Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha[\text{TD Target}] \\
Q(s,a) \leftarrow Q(s,a) + \alpha [\text{TD Target} - Q(s,a)]
$$

The term $\text{TD Target} - Q(s,a)$ is called Temporal Difference (TD). The name comes from the fact that we find a difference between our estimate of future vs our estimate of present.

This iterative averaging process allows the estimate $Q(s,a)$ to converge towards the true expected value $Q^*(s,a)$ despite the noise in individual TD targets, under appropriate conditions for the learning rate $\alpha$.

#### The Exploration vs. Exploitation Dilemma

Now, we know how to update our $Q(s,a)$ estimate once we have an experience tuple $<s, a, r, s'>$. But how should the agent choose the action $a$ in state $s$ to generate that experience in the first place?

This leads to a fundamental dilemma in reinforcement learning: Exploration vs. Exploitation.

**Exploitation**: Acting based on the current knowledge to maximize immediate expected reward. This means choosing the action $a$ that currently has the highest estimated Q-value: $a = \arg\max_{a'} Q(s, a')$. This is often called acting greedily. And it totally makes sense, if we know that taking a certain action $a$ is the best option we have, we don't we choose it? This is the essence of exploitation.

**Exploration**: Trying out actions that don't currently look like the best option. This might involve choosing an action $a \neq \arg\max_{a'} Q(s, a')$. This is necessary to discover potentially better actions whose values might initially be underestimated, or simply to get more accurate estimates for all actions.

The Dilemma: If the agent only exploits, it might get stuck in a suboptimal routine because it never tries actions that could potentially lead to much higher long-term rewards. If the agent only explores (e.g., acts randomly), it performs poorly because it never leverages the knowledge it has gained. Finding a good balance is critical for effective learning.

#### Why Exploration is Crucial for Q-Learning:

Q-learning aims to find the optimal Q-values, $Q^*(s,a)$. Its update rule uses the $\max_{a'} Q(s', a')$ term, which inherently learns about the value of acting greedily in the future. However, to ensure that the Q-values themselves converge correctly for all relevant state-action pairs, the agent needs to actually visit those pairs.

If the agent acts purely greedily from the start based on potentially poor initial Q-value estimates, it might never execute certain actions in certain states. Consequently, the Q-values for those untried actions would never be updated via the Q-learning rule, and the agent might never realize they were actually better.

Exploration, especially early in training when Q-value estimates are inaccurate, is essential to gather information across all possible actions and ensure that the estimates can eventually converge to their true optimal values.

#### The Epsilon-Greedy Policy

A simple and widely used strategy to balance exploration and exploitation during learning is the **\epsilon$-greedy policy**:

Choose a small value for $\epsilon$ (epsilon), typically between 0 and 1 (e.g., 0.1).

With probability $1 - \epsilon$: Exploit by choosing the action with the highest current Q-value: $a = \arg\max_{a'} Q(s, a')$.

With probability $\epsilon$: Explore by choosing an action uniformly at random from all possible actions available in state $s$.

Role of $\epsilon$:

If $\epsilon = 0$, the policy is purely greedy (pure exploitation).

If $\epsilon = 1$, the policy is purely random (pure exploration).

For $0 < \epsilon < 1$, the agent mostly exploits but occasionally takes a random exploratory step.

Common Practice: Often, $\epsilon$ is started at a higher value (e.g., 1.0 or 0.5) early in training to encourage broad exploration and then gradually decreased (annealed) over time towards a small value (e.g., 0.1 or 0.01). This shifts the balance from exploration towards exploitation as the agent gains more experience and its Q-value estimates become more reliable.

Using an $\epsilon$-greedy policy (or other exploration strategies) to generate the $<s, a, r, s'>$ tuples allows Q-learning to effectively learn the optimal $Q^*$ values even though its update rule focuses on the greedy path via the max operator.

And that is the complete algorithm, again the pseudo-code:

1. Start with all our estimates being zero $` \forall a,s \in A, s \in S; \space Q(s,a) = 0`$
2. Get initial $s$ from environment
2. Repeat:
    2.1. Choose $a$ from $s$ using an $\epsilon$-greedy policy
    2.2. Act in the environment and get reward $r$ and new state $s'$
    2.3. Update Q-value for this state-action pair 
    2.4. Set $s$ to $s'$

## SARSA

SARSA is also another well-known TD algorithm. It works very similar to Q-learning with just one small difference. In Q-learning our $\text{TD Target}$ was estimated as reward + maixmum Q-value of the next state. Taking maximum can be tricky, although the highest Q-value means the better action, but as we just guess (estimate) Q-values, we sometimes can be wrong in our estimations.

So why can taking the maximum be bad?

1. Overestimation Bias:
The main issue is that the $\max$ operator tends to overestimate the true value when the Q-values are only estimates (and thus noisy). If the Q-values for the next state $s'$ are not accurate (which is common during learning), the maximum is likely to select not just the best action, but also the action whose Q-value has been overestimated due to random noise or limited experience. Over time, this can cause the learned Q-values to systematically overestimate the true values.

2. Propagation of Errors:
Because Q-learning always updates towards the maximum, any overestimated Q-value can propagate through the value function, making the problem worse as learning progresses.

3. Noisy Environments:
In stochastic or noisy environments, the $\max$ operator can amplify the effect of random high rewards, further increasing the overestimation.

SARSA, in contrast, uses the Q-value of the actual action taken in the next state (following the current policy, often $\epsilon$-greedy), not the maximum. Its TD target is:

$$ \text{TD Target}_{\text{SARSA}} = r + \gamma Q(s', a') $$

where $a'$ is the action actually chosen in $s'$. This makes SARSA more conservative and less prone to overestimation, especially in noisy environments. So, SARSA stands for the tuple that we gather: $<s, a, r, s`, a`>$

So, the pseudo-code is:

1. Start with all our estimates being zero $` \forall a,s \in A, s \in S; \space Q(s,a) = 0`$
2. Get initial $s$ from environment
2. Repeat:
    2.1. Choose $a$ from $s$ using an $\epsilon$-greedy policy
    2.2. Act in the environment and get reward $r$ and new state $s'$
    2.3. Choose $a'$ from $s'$ using same policy
    2.3. Update Q-value for this state-action pair 
    2.4. Set $s$ to $s'$

## Variants of SARSA

There are various variants of SARSA, one of them for example is Expected Value SARSA. This one instead of using Q-value of the (next state, next action) pair uses the expected value of the next state. The expected value is derived using the formula we already know:

$$ \text{V(s')} = \sum_{a'} \pi(a'|s') Q(s', a') $$

So, the update rule is:

$$
 Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a)] $$


How do we compute probabilities of next actions in the next state? It actually depends on our exploration strategy. Let's review the case with epsilon-greedy:

To pick some non-optimal action two conditions must happen:

1. We choose non-greedy action 
2. At random we chose this specific action

or 

$$
p(a \neq \arg\max_a Q(s, a)) = \epsilon * \frac{1}{N}
$$

where N is the total number of actions. So, to choose optimal action we either pick it greedily with probability of (1 - $\epsilon$) or we pick at random when we explore:

$$
p(a = \arg\max_a Q(s, a)) = 1 - \epsilon + \epsilon * \frac{1}{N}
$$

As you can see total probability sum to 1.

## N-step TD Learning, TD($\lambda$) and eligibility traces

So far, we've seen two main approaches for model-free value estimation:

1. Monte Carlo (MC): Waits until the end of an episode to calculate the full observed return $G_t$ and uses that to update $Q(s_t, a_t)$. It's unbiased but can have high variance and only learns offline (after the episode).
2. One-step Temporal Difference (TD): Updates $Q(s_t, a_t)$ immediately after one step, using the observed reward $r_{t+1}$ and the current estimate of the next state-action value $Q(s_{t+1}, a_{t+1})$(like in SARSA) or $max_aQ(s_{t+1}, a)$ (like in Q-learning). This is called bootstrapping. It has lower variance but can be biased by its own estimates and learns online.

These two methods represent extremes. MC uses the full actual return, while one-step TD uses only one step of actual reward and then bootstraps. Can we find a middle ground? Yes, and that's where N-step TD learning comes in.

### N-Step TD Learning

The core idea of N-step TD learning is to look ahead $n$ steps and use the sum of actual rewards received plus the estimated value of the state reached $n$ steps later.

The n-step return $G_{t:t+n}$ at time $t$ is defined as:

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$

where $V(S_{t+n})$ is the estimated value of state $S_{t+n}$. 

If $t+n \ge T$ (where $T$ is the terminal time step of the episode), then all rewards from $R_{t+n}$ onwards are actual rewards until termination, and the $V(S_{t+n})$ term is set to 0 (as the value of a terminal state is 0). In this case, $G_{t:t+n}$ becomes the full Monte Carlo return $G_t$.

The Q-value (or state-value) update then uses this n-step return as the target:

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[G_{t:t+n} - Q(S_t,A_t)]$

(If adapting for Q-learning like updates, the term $V(S_{t+n})$ in the n-step return would be $\max_a Q(S_{t+n}, a)$. For SARSA-like n-step, it would be $Q(S_{t+n}, A_{t+n})$ if $S_{t+n}$ is not terminal.)

#### Understanding N-Step TD: Spectrum of Methods:

- If $n = 1$, then the 1-step return for SARSA is $G_{t:t+1} = R_{t+1} + \gamma Q(S_{t+1},A_{t+1})$ (assuming $S_{t+1}$ is not terminal), which makes the update exactly the one-step SARSA update target.

- If $n$ is very large (e.g., $n \ge T-t$, so it reaches the end of the episode), then $G_{t:t+n}$ becomes the full Monte Carlo return $G_{t}$.

#### Bias-Variance Trade-off:

- Larger n: The target $G_{t:t+n}$ relies less on bootstrapped estimates (like $Q(S_{t+n}, A_{t+n})$) and more on actual experienced rewards, leading to lower bias (as it's closer to the true definition of value). However, because it depends on a longer sequence of potentially random actions and rewards, its variance is higher.

- Smaller n: The target relies more on bootstrapped estimates (which can be biased if the current Q-values are inaccurate), but has lower variance as it depends on fewer random events.

#### Implementation 

To perform an update for $Q(S_t,A_t)$ using an n-step return, the agent needs to store the last $n$ rewards and states (and actions if n-step SARSA). This means the update for the experience at time $t$ is delayed until time $t+n$ when $R_{t+n}$ and $S_{t+n}$ (and $A_{t+n}$) are observed.

N-step methods often perform better than pure one-step TD or pure MC because they can find a "sweet spot" in this bias-variance trade-off. The optimal value of $n$ is problem-dependent.

### TD($\lambda$) and Eligibility Traces

While N-step TD allows us to choose how far we look into the future, it raises the question: which $n$ is best? And can we somehow combine the benefits of looking at multiple n-step returns simultaneously? 

This is the motivation behind TD($\lambda$). TD($\lambda$) elegantly averages many different n-step returns. Instead of picking one $n$, it considers all possible n-step returns, weighting each n-step return $G_{t:t+n}$ using powers of $\lambda^{n-1}$ (where $\lambda \in [0,1]$ is a new hyperparameter). This leads to the concept of the $\lambda$-return, $G_t^\lambda$.

Conceptually (this is the forward view):

$G_t^\lambda \doteq (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t$

- If $\lambda=0$, the $\lambda$-return simplifies to $G_{t:t+1}$ (the one-step TD target, as only the $n=1$ term with $\lambda^0$ survives the $(1-\lambda)$ scaling).

- If $\lambda=1$, the $\lambda$-return simplifies to $G_t$ (the full Monte Carlo return, as all weight goes to the final term).

While the $\lambda$-return provides a sophisticated theoretical target, calculating it directly using this forward view (looking at all future n-step returns) can be complex to implement and inefficient for online learning, as it requires waiting until the end of an episode or at least many steps into the future. This is where **eligibility traces** come in.

### Eligibility Traces: The Backward View

Eligibility traces provide a practical and computationally efficient way to approximate or exactly implement TD($\lambda$) updates. They offer a "backward view" mechanism.

**What is an eligibility trace?**

An eligibility trace, denoted $E_t(s,a)$ at time $t$ for a state-action pair $(s,a)$, is a temporary record. It tracks how "eligible" a pair is for learning updates based on how recently and (for accumulating traces) frequently it was visited *within the current episode*. Pairs visited more recently are more "eligible" to receive credit or blame for future TD errors.

#### How Eligibility Traces Work (e.g., for SARSA($\lambda$)):

Initialize eligibility traces $E(s,a) = 0$ for all $(s,a)$ at the beginning of each episode.

At each time step $t$, after taking action $A_t$ in state $S_t$ and observing $R_{t+1},S_{t+1},A_{t+1}$ (or just $R_{t+1}, S_{t+1}$ if $S_{t+1}$ is terminal):

a. Calculate the one-step TD error: $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$ (If $S_{t+1}$ is terminal, $Q(S_{t+1}, A_{t+1})$ is 0).

b. Increment the eligibility trace for the current state-action pair $(S_t, A_t)$. A common way is: $E(S_t, A_t) \leftarrow E(S_t, A_t) + 1$ (This is called an accumulating trace). (Another type, replacing trace, sets $E(S_t, A_t) \leftarrow 1$ if this pair was just visited, or if $Q(S_t, A_t)$ was maximal for $S_t$ in some Q($\lambda$) variants).

c. Update the Q-values for all state-action pairs $(s,a)$ based on this single TD error $\delta_t$ and their current eligibility trace $E(s,a)$: 

$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E(s,a) \quad \text{(for all s,a)}$

d. Decay all eligibility traces for the next step: $E(s,a) \leftarrow \gamma \lambda E(s,a) \quad \text{(for all s,a)}$

Intuition:

When a TD error $\delta_t$ occurs (e.g., an unexpected reward), eligibility traces "broadcast" this learning signal back to all recently visited state-action pairs. The strength of the update to a past pair $(s,a)$ depends on how large $\delta_t$ is and how high its current eligibility trace $E(s,a)$ is. The trace $E(s,a)$ is higher if $(s,a)$ was visited very recently, or multiple times recently (for accumulating traces), and it decays exponentially with the factor $\gamma \lambda$ for each step that passes where $(s,a)$ is not visited.

#### SARSA($\lambda$) Pseudo-code (with accumulating traces):

Initialize $Q(s,a)$ arbitrarily (e.g., to 0) for all $s,a$.

Choose $\alpha, \lambda, \gamma$.

For each episode:

Initialize $E(s,a)=0$ for all $s,a$.

Choose action $A$ from initial state $S$ using policy derived from Q (e.g., $\epsilon$-greedy).

Loop for each step of episode (while $S$ is not terminal): 

a. Take action $A$, observe reward $R$ and next state $S'$. 

b. If $S'$ is terminal: i. $\delta \leftarrow R - Q(S,A)$. 
 ii. Set $A'$ to a dummy action (as there's no next action). 

c. Else ($S'$ is not terminal): i. Choose $A'$ from $S'$ using policy derived from Q (e.g., $\epsilon$-greedy). ii. $\delta \leftarrow R + \gamma Q(S',A') - Q(S,A)$. d. Increment trace for current pair: $E(S,A) \leftarrow E(S,A) + 1$. e. For all state-action pairs $(s_{all}, a_{all})$: i. $Q(s_{all}, a_{all}) \leftarrow Q(s_{all}, a_{all}) + \alpha \delta E(s_{all}, a_{all})$. ii. $E(s_{all}, a_{all}) \leftarrow \gamma \lambda E(s_{all}, a_{all})$. f. If $S'$ is terminal, break loop. g. $S \leftarrow S'; A \leftarrow A'$. #### Benefits and Drawbacks of TD($\lambda$):
Benefits:

Often significantly speeds up learning and improves performance compared to one-step methods (TD(0)) or pure Monte Carlo methods, especially when rewards are delayed or sparse. It effectively bridges the gap towards MC's use of full returns.
Provides a flexible mechanism to balance the bias (from bootstrapping too early) and variance (from long MC returns) through the $\lambda$ parameter.
Drawbacks:

Adds another hyperparameter, $\lambda$, which needs to be tuned.
Can be more computationally intensive per step if implemented naively by explicitly looping through and updating all Q-values and traces (as in the tabular pseudocode). For function approximation, the trace vector has the same size as the parameter vector, making this more efficient.
In essence, N-step TD learning generalizes one-step TD and MC by choosing a fixed lookahead $n.TD(\lambda$) further refines this by elegantly averaging across (conceptually) all possible n-step returns, with eligibility traces providing an efficient online algorithmic mechanism to achieve this powerful backward credit assignment. These methods are powerful tools in the model-free reinforcement learning arsenal.