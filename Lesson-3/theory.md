# Introduction to RL. Part 3. Deep Reinforcement Learning

Welcome to the third part of this introductory course! I hope last two lessons were enjoyable and knowledgable! 

So, let's recap. In the previous lesson, we have studied the basics of Reinforcement Learning (RL) including $V(s)$, $Q(s,a)$ and their optimal versions. We have also studied how to estimate them using Monte Carlo and Temporal Difference (TD) methods. We reviewed the two most well-known algorithms Q-learning and SARSA.

Nevertheless, you might probably noticed that the problems that we solved were quite simple. And yet you also probably heard about RL playing video-games at human-level and even beating humans at them. So, how is it possible with what we learned so far? 

If you remember when we were discussing major problems on why Value Iteration is not suitable for modern RL problems we identified two main issues:
- Need for the model of the environment
- Only applicable to problems with small state space

Basically, the TD and MC algorithms are working well with small problems, by small I mean that the problem has finite and more importantly small state-action space. We can store all the pair in a table and look at each row and column to get the Q-value. But as the problem get bigger the state-space gets bigger as well. We conclude to the point that no computer in the world can hold that big table. So, we need to find a way how we can approximate Q-value without storing it somewhere.

We have seen something similar in our first lesson when we used neural networks (CEM, ES) to approximate our policy, and a short sneak peak at the following content: we will ask neural networks to help us.

## Table of Contents

1. [Deep Learning](#deep-learning)
2. [Deep Q-Learning](#deep-q-learning)
3. [Problems with Deep Q-Learning](#problems-with-deep-q-learning)
4. [Deep Q-Network (DQN)](#deep-q-network-dqn)
5. [Advancements of DQN](#advancements-of-dqn)
   - [Overestimation Problem and Double DQN](#overestimation-problem-and-double-dqn)
   - [Double Deep Q-Network (DDQN)](#double-deep-q-network-ddqn)
   - [Prioritized Experience Replay](#prioritized-experience-replay)
   - [Dueling DQN](#dueling-dqn)
   - [N-step Return](#n-step-return)
6. [Other Advancements](#other-advancements)
   - [RAINBOW: The Ultimate Combination](#rainbow-the-ultimate-combination)
   - [Beyond RAINBOW: The Continuing Evolution](#beyond-rainbow-the-continuing-evolution)
7. [Conclusion](#conclusion)


## Deep Learning



Thanks to the advancements in Deep Learning (DL), people started looking at the conjunction between RL and DL. The fundamental question was: if neural networks are so powerful, can they also approximate Q-values? Basically, let the network take as input state representation and output Q-values for all possible actions. And the short answer: they can *but with many hacks*!

--- 

**A quick important note.**

At first glance it might be intuitive to represent this problem as a classification problem. Given state $s$ we want to get an index of the most profitable action $a$, so the network outputs the probability distribution over actions, i.e. let the neural network be our policy. After all we did the same in ES and CEM examples.

However, to use any TD-learning update rules we need to estimate state or state-action values. If the network only outputs the probability distribution over all actions, how can we use that to estimate the Q-values?

Moreover, how can we decide which actions are the best if we do not have any estimates of Q-values? Remember that our rule was to pick action corresponding to the highest value, so we need an estimate for them. 

Basically, what I am trying to say that it makes much more sense to let the network output not a probability distribution over action space, but do a regression task. Given the state $s$ as the input, the network outputs the estimates of Q-values for each action. As these values are real numbers, this task is inherently of a regression type. 


## Deep Q-Learning

Okay, let's modify a bit our notion of Q-values to $Q(s, a; \theta)$ where $\theta$ are parameters of the network (read it as Q-values obtained from the neural network with parameters $\theta$). We can think of it as a function that takes state $s$ as input and outputs Q-value for all actions. The idea is to train the network to approximate $Q(s, a)$ for all possible actions. 

Well, we cannot just collect the dataset of $(s, [Q(s,a) \  \text{for each} \  a  \in A])$ in the same way as we collected $(s,a)$ pairs in the Cross-Entropy Method, because we simply do not know the true Q-values. So we have to think of a workaround, and you know what, in the previous lesson we already established these workarounds. More specifically, the Temporal Difference or Monte-Carlo learning is exactly what we need. That means that we train the network in the same way we trained vanilla Q-learning!

So, our Q-learning update rule is:

$$
\delta = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t) - Q(s_t, a_t; \theta_t) \\
Q(s_t, a_t; \theta_t) \leftarrow Q(s_t, a_t; \theta_t) + \alpha \delta
$$


But we need to slightly modify it to work correctly with neural networks, because we do not store these Q-values. After all we are not updating the entries in the table anymore, now we are updating the weights and biases of the neural network that estimates them. So, naturally, we need to define the loss function. We pick a simple, yet effective L2 (MSE) loss:

$$
L(\theta) = \frac{1}{2} \left( Q^*(s_t, a_t) - Q(s_t, a_t; \theta_t) \right)^2 \\ 
Q^*(s_t, a_t) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t)
$$

This loss represents an Euclidean distance between two points in the high-dimensional state-action space. First point is our current estimate of Q-value for some $(s, a)$ coordinate. And the second point is the updated estimate using newly generated data sample. By trying to minimize this distance we are moving our estimate towards the new updated estimate. And by doing it iteratively, we hope to converge towards the optimal Q-values.

So the intuition behind Deep Q-Learning is exactly the same as in vanilla Q-learning. We play the episode, at each time step we collect `<s, a, r, s'>` samples. We calculate the new estimate for $Q^*(s,a)$ using newly experienced data. Our updated Q-value is then compared with what the neural network has estimated via L2 loss. Then this loss is backpropagated to update the weights, such that in the future the network would improve its estimate towards the updated version. And this is repeated many times.

So, this is the first approach and you know what? It did not work! I know I know; so many words and it does not work in the end. Let's think why?

## Problems with Deep Q-Learning

Okay, let's start with the first problem that is not that obvious but very serious. Every deep learning problem has one assumption that must be valid. It is called i.i.d. or independent and identically distributed data. Basically saying that the data samples the network is using to learn from are independent from one another. But in our case as we iterate in the environment our data coming back from it is at least temporally correlated, right? Doing action $a_t$ in state $s_t$ will affect the next data sample, because that action itself is the main source of the next state $s_{t+1}$. So, there is a temporal correlation between samples ($s_{t+1}$, $a_{t+1}$) and ($s_t$, $a_t$).

Consider this: imagine doing the same action (*Go Left*) from the initial state (entry to the maze) leading to the left part of the maze several times in a row; these sequences of experiences will be very similar. Training a neural network directly on such consecutive, correlated samples is inefficient and can lead to unstable learning. So, we have to deal with it somehow. 

The second major problem is "chasing own tail". Look at the update rule for our network:

$$
L(\theta) = \frac{1}{2} \left( Q^*(s_t, a_t) - Q(s_t, a_t; \theta_t) \right)^2 \\ 
Q^*(s_t, a_t) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t)
$$

So, our network tries to make the estimate of $Q(s_t, a_t)$ closer to $Q^*(s_t, a_t)$. But look carefully at the equation of $Q^*(s_t, a_t)$, to calculate this value we are using the estimates of the **same** network but of the next state: $\max_{a'} Q(s_{t+1}, a'; \theta_t)$. The neural network is not that all-mighty, changing params to better estimate $Q(s_t, a_t)$ will affect estimation of $Q(s_{t+1}, a^*)$. It is like a dog chasing its own tail. If you question why we were not seeing this issue in Q-learning, the explanation is simple. Changing one entry in the table does not affect another entry, but when we are using the same network for both estimates, then we are screwed. 

So, we have two problems that require innovation. The first problem was actually easily solved, instead of immediately updating the neural network with newly gathered `<s, a, r, s'>` sample we instead store it in memory. After enough samples are collected, we are going to sample a random batch of experience and update the network on this random batch. This way we are not learning from consecutive samples, but from the random samples of our experience. This destroys the temporal correlation and solves our problem. That storing the experience approach was called **Replay Buffer**. In practice it is often implemented as a double-ended queue of some maximum size N to keep only the last N samples. Also, by sampling a batch we are doing mini-batch gradient descent which was shown as an improvement over full-batch gradient descent.

The second problem required a bit more work, but still the solution was simple. The main idea behind the solution was that instead of having one network for two estimates, we will have two networks, one for each estimate. The first network estimates $Q(s_t, a_t)$ and the second network estimates $Q(s_{t+1}, a^*)$. So, basically we decouple the correlation between online estimate $Q(s_t, a_t)$ and target estimate $Q(s_{t+1}, a^*)$. This decoupling stabilizes training. In practice, the first network is called an online network and the second is called a target network. Both of them have exactly the same architecture. During the update, we are updating only the weights of the online network and do not change the target network. But to keep the target estimates up-to-date, we periodically copy weights from the online to the target network.

And that's it, these two innovations resulted in a beautiful paper that was able to learn how to play a variety of Atari games just from pixels! There was no pre-defined policy of how to act when someone is shooting the agent, it learned how to interpret the pixels into Q-values to make the best move! In some games the agent could even surpass humans!

## Deep Q-Network (DQN)

So, the pseudo-code for DQN is

1. Initialize target and online networks
2. Copy params from online network into target network: $\theta_{target} \leftarrow \theta_{online}$
3. Initialize a replay buffer $D$ with some size $L$
4. Get initial $s_t$ from the environment
5. repeat each episode $e$ 
   1. Choose action $a_t$ given $s_t$ and our $\epsilon$-greedy policy
   2. Apply action $a_t$ and obtain <$r_{t+1}, s_{t+1}, \text{done}$> where $\text{done}$ is a boolean flag to check if the next state is terminal
   3. Store $<s_t, a_t, r_{t+1}, \text{done}, s_{t+1}>$ in $D$
   4. If the current size of $D$ is larger than batch size: Sample $B$ batch from $D$ 
        1. Compute TD-target for all samples $j \in B$
            - $y_j = r^j_{t+1} + \gamma \max_{a^j_{t+1}} Q(s^j_{t+1}, a^j_{t+1}; \theta_{target})$       if not $\text{done}$
            - $y_j = r^j_{t+1}$  if $\text{done}$
        2. Compute loss and apply gradient descent
            - $L = 1/B \cdot \sum_j (y_j - Q(s_j, a_j, \theta_{online}))^2$
   5. Set $s_t = s_{t+1}$

Looks quite simple right? Because in fact it is. But don't be illusioned by its simplicity, this algorithm was able to surpass humans in some games. Furthermore, it was a first example of Reinforcement Learning being able to deal with high-dimensional state space problems. This led to many other advancements in the field.

## Advancements of DQN

This paper was introduced in 2013, so naturally people started to improve it. Here I will mention some advancements I think that are interesting. There are others that are omitted here, but I think this lesson gives you enough of foundation to understand them on your own :)

### Overestimation Problem and Double DQN
The two-network scheme (online and target networks) solved the "chasing its own tail" problem by providing a stable target for our updates. However, another subtle but significant issue remained: **overestimation bias**.

We've seen this before in Q-learning. When we use a max operator on a set of noisy estimates, we are likely to pick a Q-value that is high due to random error than a true highest Q-value. This means our algorithm can become overly optimistic, systematically overestimating the true value of states.

This problem was even more amplified in DQN. In tabular Q-learning, we could eventually learn the true value for every state-action pair. But a neural network can't memorize everything; it has to generalize and approximate. Especially early in training, these approximations are very noisy. When we use a target calculation like this:

$$
y_j = r_{j+1} + \gamma \max_{a'} Q(s_{j+1}, a'; \theta_{target})
$$

The max operator becomes a major problem. It scans over all possible next actions and, because the $Q_{target}$ is still learning and noisy, it's likely to find an action whose value is high simply due to estimation error. Using this inflated value as the target means our online network is consistently learning from an overly optimistic signal. This can lead to poor policies and unstable training.


### Double Deep Q-Network (DDQN)

Double DQN, introduced by Hado van Hasselt et al., provides a simple yet brilliant fix to this overestimation problem. The core idea is to **decouple the action selection from the action evaluation** when calculating the target value.

Instead of using the target network for both jobs, DDQN splits the responsibility:

- **Action Selection:** First, we use our up-to-date online network to ask, "For the next state $s_{j+1}$, which action do you think is the best?"

  $$
  a^*_{online} = \arg\max_{a'} Q(s_{j+1}, a'; \theta_{online})
  $$

- **Action Evaluation:** Then, we take that chosen action, $a^*_{online}$, and ask our stable target network, "What is the value of that specific action?"

  $$
  Q(s_{j+1}, a^*_{online}; \theta_{target})
  $$

So, the new DDQN target $y_j$ becomes:

$$
y_j = r_{j+1} + \gamma Q(s_{j+1}, \arg\max_{a'} Q(s_{j+1}, a'; \theta_{online}); \theta_{target})
$$

(As before, $y_j = r_{j+1}$ if $s_{j+1}$ is a terminal state).

---

**Why does this work?** The online network might still pick an action whose value it overestimates. However, **it's much less likely** that the separate, slightly older target network will also happen to overestimate the value of that same specific action. By separating selection and evaluation across the two networks, we break the feedback loop that created the optimism. This leads to more accurate Q-value estimates and, as a result, more stable and better-performing agents.

This small change to the target calculation often provides a significant performance boost and has become a standard technique in modern value-based reinforcement learning.

---

If you remember, in Q-learning, that problem was a motivation for exploring other update rules and coming to SARSA. As we are just reviewing the history of innovations of DQN, I cannot accurately tell you why they have not tried using SARSA or EV-SARSA update rule to try to solve the overestimation problem. What I can do is provide my opinion:

DQN uses a Replay Buffer, which means that the approach is heavily off-policy. The sample that we collect from the Replay Buffer - `<s, a, r, s'>` - may be generated by a very old version of our network and the current network can be several update steps ahead of it.

SARSA, on the other hand, is more on-policy. It evaluates what next action the current policy would have taken, and then uses its Q-value for update. As we had to use a Replay Buffer to break temporal correlation, then it means that we would have needed to store `<s, a, r, s', a'>` samples. Updating our current network (which might be hundreds of update steps ahead) with the sample generated by some older network (which can be hundreds of update steps behind the current network) can be bad. Think of it like you being an adult asking advice from yourself when you were two years old.

Using an up-to-date estimate of Q-value for the next state and the best action is, in my opinion, a much more stable and better approach.

---

### Prioritized Experience Replay

Okay, the next improvement was not directly affecting DQN, but rather improving the overall training process. We said the batch of experience is sampled randomly from the replay buffer. Choosing randomly breaks the temporal correlation and stabilizes the training, but maybe we can at the same time break temporal correlation and leverage some prior knowledge about our samples to sample smarter? Of course we can!

This is what was introduced in Prioritized Experience Replay (PER) paper, the authors showed that picking not at random but with some pre-defined logic results in a better learning curve. 

#### What and how do we use for sampling?

Okay, when I told you that we use some pre-defined logic I did not mean that we research the problem and then instruct the agent to do something specific. I meant that we can leverage something that the agent itself is calculating to improve our sample logic. For each tuple of experience `<s, a, r, done, s'>` there is a corresponding TD error (easily computed). This TD error can also be understood as a measure of surprise for the network. Think of it: the bigger the TD error, the more the network was wrong about the estimation, meaning the more the network was surprised. So, naturally, we would want to train on this sample more, to make the surprise much less.

This intuition is exactly what PER formalizes. Here's how it generally works:

1. Assigning Priorities:

When a new experience (transition) `<s, a, r, done, s'>` is added to the replay buffer, its TD error isn't known yet. So, it's typically given a maximum priority initially to ensure it gets sampled at least once.

When a transition $i$ is sampled from the buffer and used for learning, its TD error $\delta_i$ is calculated (as we did in the DQN update: $\delta_i = y_i - Q_{online}(s_i, a_i)$).

The new priority $p_i$ of this transition is then set based on this TD error, usually using its absolute value plus a small positive constant $\epsilon_{PER}$ (to prevent any transition from having zero probability of being sampled): $p_i = |\delta_i| + \epsilon_{PER}$

This priority is further raised to a power $\alpha$ (a hyperparameter, where $0 \le \alpha \le 1$) to control the degree of prioritization: $p_i^\alpha$. If $\alpha=0$, we get uniform random sampling. If $\alpha=1$, we get full prioritization based on $p_i$.

2. Sampling Based on Priorities:

Instead of sampling uniformly, transitions are now sampled proportionally to their assigned priorities - $p_i^\alpha$. The probability of sampling transition $i$ becomes: 

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

This means transitions with higher priorities (larger TD errors) are more likely to be selected for the training mini-batch.

This is typically implemented efficiently using a specialized data structure called a SumTree (as seen in the OpenAI Baselines code). A SumTree allows for both efficient updating of priorities and efficient sampling according to these priorities, usually in $O(\log N)$ time, where N is the buffer size.

3. The Bias Problem and Importance Sampling (IS) Correction:

Unfortunately, this trick has its own pitfalls. Sampling transitions non-uniformly like this introduces a bias. The network sees "surprising" or high-error samples more frequently than they actually occurred in the agent's interaction with the environment. This can distort the learning process because the distribution of samples used for updates no longer matches the true distribution of experiences. If uncorrected, this can lead the Q-values to converge to incorrect values.

The Solution: **Importance Sampling (IS) Weights** 

Importance Sampling is a method used to address a common statistical problem: how to evaluate one strategy using data that was generated by a different one. In reinforcement learning, this is central to off-policy learning, where an agent needs to learn about a new, improved policy using experiences gathered from older, different policies.

The core mechanism is to re-weight each data point to correct for this mismatch. For every experience, a correction factor known as the importance ratio is calculated. This ratio compares the probability of taking a certain action under the new target policy to the probability of taking that same action under the old behavior policy that originally generated the data.

An experience is given a higher weight if its action is more probable under the new policy, increasing its influence on the learning update. Conversely, its influence is down-weighted if the action is now less probable. This process mathematically corrects for the bias of using off-policy data, allowing the agent to make an accurate judgment about its new strategy.

The main benefit of this technique is a significant improvement in sample efficiency. It enables the agent to continuously and safely learn from its entire history in the replay buffer, even as its strategy changes. This ability to effectively reuse past experiences is a key component of many modern reinforcement learning algorithms.

For each sampled transition $i$, an IS weight $w_i$ is calculated to adjust its contribution to the gradient update. The weight is typically: 
$$
w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta
$$

where:

- $N$ is the current size of the replay buffer.
- $P(i)$ is the probability with which transition $i$ was sampled (calculated above).
- $\beta$ is another hyperparameter ($0 \le \beta \le 1$) that controls how much correction is applied. 

    It is usually annealed (linearly increased) from an initial value (e.g., 0.4) towards 1 over the course of training. Starting with a smaller $\beta$ helps stabilize learning early on when Q-estimates are very noisy. This is because, at the beginning of training, the Q-values predicted by the network are highly inaccurate and unstable, so applying full importance sampling correction (i.e., $\beta=1$) can amplify the noise and lead to unstable updates. By starting with a lower $\beta$, the effect of the importance sampling weights is reduced, allowing the learning process to be more robust to the initial inaccuracies. As training progresses and the Q-value estimates become more accurate, $\beta$ is gradually increased to 1 to fully correct for the bias introduced by prioritized sampling.

These weights $w_i$ are then used to scale the loss for each transition in the mini-batch. For example, if the loss for sample $i$ is $L_i = \delta_i^2$, the weighted loss becomes $w_i \cdot \delta_i^2$. 

The gradient update for that sample is effectively scaled by $w_i$. 

$$
\text{Weighted Loss} = \frac{1}{\text{BatchSize}} \sum_i w_i \cdot \delta_i^2
$$ 

This down-weights the updates for transitions that were sampled very frequently (high $P(i)$, low $w_i$) and up-weights those that were sampled rarely (low $P(i)$, high $w_i$), thus correcting for the biased sampling. The weights are often normalized by dividing by $\max_j w_j$ in the batch to keep update magnitudes stable.

4. Updating Priorities After Learning:

Once a transition $i$ from the mini-batch has been used for a learning update and its new TD error $\delta_i$ is known, its priority $p_i$ in the buffer must be updated to reflect this new error. This ensures that future sampling correctly prioritizes it based on its current "surprise" level.

----

I know sounds kinda complex, but the overall logic has not changed, all we have to do is to scale the loss correspondigly.

In summary, PER changes the DQN training loop like this:

1. Store new transitions in the buffer with high initial priority.

2. When sampling a batch: 

    1. Sample proportionally to $p_i^\alpha$. 
    2. Calculate IS weights $w_i$ using the current $\beta$.

3. When computing the loss for the batch, scale each sample's loss by its IS weight $w_i$.

4. After the update, update the priorities of the sampled transitions in the buffer using their new absolute TD errors.

5. Anneal $\beta$ towards 1.

By focusing on "surprising" transitions while correcting for the induced bias, PER often leads to significantly faster learning and better final performance compared to uniform sampling. It's a clever way to make the learning process more efficient!

### Dueling DQN

Okay, you remember how we justified learning Q-values instead of just $V(s)$? That was because knowing $V(s)$ alone isn't enough to pick the best action if we don't have a model of the environment (i.e., we can't easily see what action leads to what next state and its value). With $Q(s,a)$, choosing the best action is easy: just take the $\arg\max_a Q(s,a)$. So, you might wonder, why would anyone want to bring $V(s)$ back into the picture? But it was done in the Dueling DQN paper. 

The insight behind that paper is that for many states, the value of the state itself ($V(s)$) is a dominant factor, and the specific action taken might only result in minor differences in outcome. Imagine your agent is in a game state where a missile is inevitably going to hit it, no matter what action it takes. All actions from this state are "bad" because the state itself is terrible. In such cases, learning the Q-value for every single action independently is inefficient. The network would have to learn that $Q(s, \text{duck})$ is very negative, $Q(s, \text{jump})$ is very negative, and $Q(s, \text{run})$ is also very negative, largely because $V(s)$ itself is so negative. 

So, the proposal was that it's more efficient to separately estimate the state value $V(s)$ and the advantage $A(s,a)$ for each action.

- State Value $V(s)$: Represents how good or bad it is to be in state $s$ in general.
- Advantage $A(s,a)$: Represents how much better or worse taking action $a$ is compared to the other actions possible in state $s$. It's defined as: $A(s,a) = Q(s,a) - V(s)$

To compute both values, the network architecture is modified to have two separate streams (or "heads") after some initial shared layers:

- Value head outputs a single scalar value, which is an estimate of $V(s)$.

- Advantage head outputs a vector of values, one for each action, estimating $A(s,a)$ for each action.

I want to emphasize here that the network is still estimating Q-values, we just change the way we calculate them. More specifically, by knowing both $V(s)$ and $A(s,a)$, we have enough to reconstruct $Q(s,a)$: 

$$
Q(s,a) = V(s) + A(s,a)
$$

However, imagine you added +100 to $V(s)$ and subtracted 100 from $A(s,a)$; the resulting $Q(s,a)$ did not change, right? To ensure that the value head truly learns the state value and the advantage head truly learns the advantages in a stable and unique way, a specific aggregation layer is used:

$$
Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)
$$

By subtracting the mean of the advantages, this formulation forces the advantages for a state to sum to zero. This helps $V(s)$ become a more robust estimate of the actual state value, and $A(s,a)$ represents the true relative preference for each action.

#### Why is this better?

- **More Efficient Learning** 

When the environment signals that a state $s$ is good or bad, this information can be learned more directly by the Value Stream. This single update to $V(s)$ then efficiently informs the Q-values of all actions in that state. The network doesn't have to redundantly learn the common state value component for each action's Q-value output.

- **Better Generalization**

The network can learn a good estimate of $V(s)$ even if not all actions have been frequently tried in state $s$. This stable $V(s)$ baseline then helps in evaluating actions more reliably.

- **Improved Performance** 

This factored representation often leads to faster learning and better final policy performance, especially in environments where many states have actions whose values are very similar (i.e., advantages are small).

In conclusion, Dueling DQN doesn't get rid of Q-values; it provides a more intelligent internal structure for the neural network to learn and represent them by decomposition into state values and action advantages. This architectural change can be combined with other improvements like Double DQN and Prioritized Experience Replay.

### N-step return

A final question you might ask is: if we used eligibility traces in tabular Q-learning to propagate rewards back more quickly, can we use them in DQN? The answer is **no**, and it's worth understanding exactly why.

To understand the incompatibility, we first need to recall how eligibility traces work.

#### What is an Eligibility Trace?

Think of an eligibility trace as a short-term memory or a "trail of breadcrumbs." When an agent visits a state-action pair $(s, a)$, it leaves a "trace" that marks its visit. This trace then slowly fades away over time. The purpose of this trace is **credit assignment**. When a surprising event happens later on (a high or low TD-error), the credit (or blame) for that surprise is propagated backward along this trail. The more recent a visit, the stronger its trace, and the more credit it receives. This allows a single TD-error to update not just the immediately preceding state-action pair, but the entire recent sequence of pairs that led to it.

#### The Core Requirement: A Contiguous Trajectory

For this backward propagation of credit to work, the trace must be **unbroken**. The algorithm needs to know the exact, contiguous sequence of states and actions:

$$
\ldots \rightarrow (s_{t-2}, a_{t-2}) \rightarrow (s_{t-1}, a_{t-1}) \rightarrow (s_t, a_t)
$$

#### The Conflict with Experience Replay

The entire purpose of the replay buffer is to destroy this contiguous sequence. It shatters the temporal correlations in the agent's experience. When you sample a mini-batch from the buffer, you get a collection of completely unrelated transitions, for example:

- Transition 1: $(s_{58}, a_{58}, r_{59}, s_{59})$ from episode 3
- Transition 2: $(s_{102}, a_{102}, r_{103}, s_{103})$ from episode 7
- Transition 3: $(s_{23}, a_{23}, r_{24}, s_{24})$ from episode 1

When you calculate the TD-error for Transition 1, you have no way of knowing what state-action pair came before it $(s_{57}, a_{57})$. The "breadcrumb trail" is gone. It's impossible to propagate the error backward along a trace that doesn't exist in the shuffled mini-batch.

This fundamental conflict between the **on-policy** nature of eligibility traces and the **off-policy** nature of experience replay is why they cannot be used together.

---

### The Alternative: N-Step Returns

However, we can achieve a similar goal—looking further into the future to create a more stable learning target—with **N-Step Returns**. Instead of a 1-step target, we can unroll the trajectory for $N$ steps:

$$
y = r_t + \gamma r_{t+1} + \ldots + \gamma^{N-1} r_{t+N-1} + \gamma^N \max_{a'} Q(s_{t+N}, a'; \theta_{\text{target}})
$$

This provides a target that incorporates more real rewards, often stabilizing and accelerating learning. This technique is perfectly compatible with a replay buffer (as long as you store N-step transitions) and is a key component of many modern algorithms, including the famous **Rainbow DQN**, which combines all of these advancements (DDQN, PER, Dueling, N-Step, etc.) into a single, powerful agent.

So, you have to modify what you store in the memory (either Replay Buffer or PER). Given N-step return you have to store a tuple consisting of:
- $s_t$ -- Start state 
- $a_t$ -- Action taken in start state
- $r_{t+n+1}$ -- Accumulated reward after taking N-steps after start state
- $\text{done}_{t+n}$ -- Whether or not the state $s_{t+n+1}$ was terminal 

Furthermore, because there can be cases where the agent hits the terminal state before all N steps are done you have to store additionally $n$ which is the number of actual steps that were taken.

## Other Advancements

The improvements we've covered in this lesson—Double DQN, Prioritized Experience Replay, Dueling DQN, and N-step returns—represent some of the most important and foundational enhancements to the original DQN algorithm. However, it's important to understand that the field of deep reinforcement learning has been incredibly active, and there are many more sophisticated improvements that have been developed.

Explaining all possible improvements to vanilla DQN would require an entire course on its own, and many of these advanced techniques build upon the fundamental concepts we've established here. The improvements presented in this lesson were chosen because they:

1. **Address core, well-understood problems** (overestimation bias, sample efficiency, architectural efficiency)
2. **Provide clear, intuitive solutions** that help build understanding of deep RL principles
3. **Serve as building blocks** for understanding more advanced techniques
4. **Have proven their value** across a wide range of environments and applications

### RAINBOW: The Ultimate Combination

One of the most notable achievements in the DQN family is **RAINBOW DQN**, introduced by Hessel et al. in 2017. RAINBOW is not a single new algorithm, but rather a careful combination of six major DQN improvements:

1. **Double DQN** - Reducing overestimation bias
2. **Prioritized Experience Replay** - Learning from important experiences
3. **Dueling DQN** - Architectural improvements separating value and advantage
4. **Multi-step Learning** - N-step returns for better target estimation
5. **Distributional RL** - Learning the full distribution of returns rather than just expected values
6. **Noisy Networks** - Replacing ε-greedy exploration with learned exploration through noisy network parameters

The remarkable thing about RAINBOW is that it demonstrated how these individual improvements could be combined synergistically. Rather than simply adding their benefits, the combination often performed better than the sum of its parts, showing the power of systematic algorithmic development.

RAINBOW achieved state-of-the-art performance on the Atari benchmark and became a standard baseline for comparing new deep RL algorithms. It represents the culmination of years of targeted improvements to the original DQN algorithm.

### Beyond RAINBOW: The Continuing Evolution

Even beyond RAINBOW, the field continues to evolve with improvements like:
- **Implicit Quantile Networks (IQN)** for better distributional learning
- **NGU (Never Give Up)** for improved exploration in sparse reward environments
- **Agent57** which achieved superhuman performance on all 57 Atari games
- Various forms of **model-based enhancements** that combine planning with value-based learning

The key insight here is that deep RL research follows a pattern of **incremental, targeted improvements**. Each advancement identifies a specific limitation (bias, variance, exploration, representation, etc.) and proposes a focused solution. Understanding the foundational improvements we've covered gives you the tools to understand and appreciate these more advanced techniques as they continue to emerge.

# Practical details

Well, look at our version of RAINBOW in the `DRL/` folder. It's formal name is N-Step Dueling Double Deep Q-Network with Prioritized Experience Replay. This beast is very powerful, but if you look inside you will see a huge amount of hyperparameters that we need to tune. Unfortunately, RL field is very brittle. Changing even one hyperparameter can result in a very different result. So, let me give some practical default values and intuition behind each one.

- **Discount factor γ**
    Discount factor is a parameter that makes the agent more or less far-sighted. It discounts future rewards by a constant, making them less valuable than immediate rewards. Usually, most RL problems show the structure of dependency of order of actions. Meaning that in the beginning doing bad actions can result in irrecoverable bad results. So, usually that factor is set to something like 0.9 or 0.99. In our implementation, we use γ = 0.99.

- **Learning rate**
    Learning rate is a hyperparameter that controls how big of an update we give to our network. The higher the update the more the network changes, but also it means that the network is less stable. The lower the update the slower is the training. Usually practitioners set the learning rate to something like 3e-4. And on top of that decrease it towards zero closer to the end of the training. In our implementation, we use 2.5e-4 for both Atari and classic control environments.

- **Memory size**
    The replay buffer size determines how many past experiences we store for training. A larger buffer provides more diverse experiences but uses more RAM and may include very old, potentially outdated experiences. For Atari games, we use 300,000 transitions (limited by RAM constraints, though 1M is often recommended). For classic control tasks, we use 10,000 transitions to avoid training on very old data that might be less relevant. Try to use 1M buffer size with DQN on CartPole. You will see that the network won't be able to produce decent results. All because it takes a lot of time to populate 1 million samples, and the network is basically using very old data sample for training.

- **Batch size**
    This determines how many experiences we sample from the replay buffer for each learning update. Larger batches provide more stable gradients but require more computation. For Atari, we use 32 (following the original DQN paper), while for classic control we use 128 for better gradient estimates with smaller replay buffers.

- **Target network update frequency**
    How often we copy weights from the main network to the target network. Too frequent updates make training unstable (moving target problem), while too infrequent updates slow learning. Practitioners found that this hard update (simple copying every N steps) is not really the best way to do it. Instead they started using soft update.
    
    **Hard Update**: θ_target = θ_online every N steps
    **Soft Update**: θ_target = τ * θ_online + (1 - τ) * θ_target every step
    
    Soft updates gradually blend the target network towards the online network using a small interpolation parameter τ (typically 0.001-0.01). This provides smoother, more stable learning compared to the abrupt changes of hard updates. In our implementation, we use τ = 0.005 for soft updates, though we also support the traditional hard update approach (every 10,000 steps for Atari, every 500 steps for classic control) for comparison.

- **Learning frequency**
    How often we perform a learning update relative to environment steps. For Atari, we learn every 4 steps to balance computation with learning progress. For classic control, we learn every step since these environments are computationally cheaper.

- **Exploration parameters**
    We use ε-greedy exploration starting at ε = 1.0 (full exploration) and decaying to ε = 0.1 for Atari or ε = 0.02 for classic control. The decay happens over 1 million steps for Atari and 10% of total training steps for classic control. This ensures sufficient exploration early in training while converging to a mostly greedy policy later.

- **Learning starts**
    Number of random steps to take before starting learning, allowing the replay buffer to fill with diverse experiences. We use 50,000 steps for Atari and 1,000 for classic control environments.

- **Prioritized Experience Replay parameters**
    For PER, we use α = 0.6 to control prioritization strength (0 = uniform sampling, 1 = full prioritization), and β annealing from 0.4 to 1.0 to correct for the bias introduced by non-uniform sampling through importance sampling weights.

The key insight is that **hyperparameter selection is environment-dependent**. Atari games require patient, stable learning with large replay buffers due to their visual complexity and sparse rewards. Classic control tasks can afford more aggressive learning with smaller buffers due to their simpler dynamics and denser reward signals. When adapting these algorithms to new environments, start with these defaults but be prepared to tune based on your specific problem characteristics.

# Conclusion

This lesson marked a pivotal transition in our reinforcement learning journey—the leap from tabular methods to the world of **Deep Reinforcement Learning**. We've witnessed how the marriage of classical RL algorithms with modern deep learning has unlocked the potential to tackle complex, high-dimensional problems that were previously impossible to solve.

## The Revolutionary Breakthrough: From Tables to Neural Networks

We began this lesson facing a fundamental limitation: classical Q-learning and SARSA work beautifully for small, discrete state spaces where we can maintain lookup tables, but they completely break down when confronted with the vast state spaces of real-world problems. The insight that **neural networks could approximate Q-values** was nothing short of revolutionary—suddenly, an agent could learn from raw pixels, high-dimensional sensor data, and complex state representations without requiring massive lookup tables.

However, this breakthrough came with its own set of formidable challenges. We discovered that naive Deep Q-Learning fails due to two critical problems:

1. **Temporal Correlation**: The sequential nature of RL data violates the i.i.d. assumption crucial for stable neural network training
2. **Moving Targets**: Using the same network for both current estimates and target values creates an unstable "chasing its own tail" scenario

## The DQN Solution: Engineering Elegance

The **Deep Q-Network (DQN)** solved these problems with two elegant engineering innovations:

- **Experience Replay Buffer**: By storing experiences and sampling random batches, DQN broke the temporal correlations that plagued naive approaches, allowing stable gradient-based learning
- **Target Networks**: By separating the network used for current estimation from the one used for target calculation, DQN stabilized the learning process and eliminated the moving target problem

These solutions were not just technical fixes—they represented a fundamental understanding of how to adapt classical RL principles to the deep learning paradigm. The result was an agent capable of learning directly from high-dimensional observations, achieving human-level performance on complex Atari games.

## The Evolution: Targeted Improvements

What followed was a beautiful example of iterative scientific progress, with each advancement addressing specific limitations:

### **Double DQN**: Conquering Overestimation Bias
The max operator in Q-learning, problematic even in tabular settings, became even more dangerous with neural network approximation. **Double DQN** elegantly solved this by decoupling action selection (using the online network) from action evaluation (using the target network), dramatically reducing the systematic overestimation that plagued vanilla DQN.

### **Prioritized Experience Replay**: Learning from Surprise
Rather than treating all experiences as equally valuable, **PER** introduced the insight that we should learn more from "surprising" experiences—those with high TD errors. By prioritizing these experiences while carefully correcting for the introduced bias through importance sampling, PER significantly improved sample efficiency.

### **Dueling DQN**: Architectural Intelligence
The insight that many states have similar values regardless of the action taken led to **Dueling DQN**'s architectural innovation. By separately estimating state values $V(s)$ and action advantages $A(s,a)$, the network could learn more efficiently, especially in environments where action choices often have similar outcomes.

### **N-step Returns: Bridging the Gap**
When eligibility traces proved incompatible with experience replay, **N-step returns** provided an elegant alternative, allowing the network to learn from multi-step experiences and achieve better bias-variance trade-offs.

## The Practical Reality: Challenges and Limitations

While these theoretical advances are impressive, we must acknowledge the practical challenges that define the current state of deep RL:

### **Hyperparameter Sensitivity**
Deep RL algorithms are notoriously sensitive to hyperparameter choices. Learning rates, buffer sizes, target network update frequencies, exploration schedules—each requires careful tuning, and what works for one environment may completely fail for another. This brittleness makes deep RL as much an art as a science.

### **Sample Inefficiency**
Despite all our innovations, deep RL agents remain voraciously data-hungry. Where humans might master a game in minutes, these agents require millions of interactions. This sample inefficiency stems from learning everything from scratch, without the benefit of prior knowledge or intuitive understanding of physics and causality.

### **The Discrete Action Ceiling**
Perhaps most fundamentally, the entire DQN framework is built around the assumption of discrete action spaces. The max operator that lies at the heart of these methods—$\max_a Q(s,a)$—simply cannot handle continuous actions like precise steering angles or torque values. This limitation creates a hard boundary that defines when we must transition to different algorithmic families.

## The Foundation for What's Next

What we've accomplished in this lesson is remarkable: we've built a complete toolkit for value-based deep reinforcement learning, from the foundational DQN to sophisticated variants that address specific challenges. Each algorithm in the DQN family represents a targeted solution to a well-understood problem, and their combination (as seen in Rainbow DQN) demonstrates the power of systematic algorithmic development.

More importantly, we've learned the **engineering principles** that make deep RL work:
- The importance of breaking correlations in training data
- The need for stable targets in value-based learning
- The power of architectural innovations that encode domain knowledge
- The careful balance between bias and variance in learning targets

These principles extend far beyond the DQN family and form the foundation for understanding more advanced methods.

## Looking Ahead: The Next Frontier

The DQN family has shown us the incredible potential of combining deep learning with reinforcement learning, but it has also revealed its limitations. The restriction to discrete actions, the sample inefficiency, and the challenges of credit assignment in complex environments point toward the need for different approaches.

This naturally leads us to the next major branch of deep RL: **policy-based methods** and **actor-critic algorithms**. These methods directly learn policies rather than value functions, opening the door to continuous control, more sample-efficient learning, and sophisticated exploration strategies.

As we prepare for this next chapter, remember that the journey from tabular Q-learning to sophisticated deep RL algorithms like DQN represents one of the most remarkable achievements in modern AI. You now understand not just how these algorithms work, but why they were designed the way they were—and that understanding will serve as a solid foundation for everything that comes next.

The transition from value-based to policy-based methods isn't about replacing what we've learned—it's about expanding our toolkit to handle an even broader range of challenges in the fascinating world of reinforcement learning.