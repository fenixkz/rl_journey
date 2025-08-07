# Introduction to RL. Part 3. Deep Reinforcement Learning

Welcome to the third part of this introductory course! I hope last two lessons were enjoyable and knowledgable! 

Before we dive in, let's recap our journey so far. In the previous lesson, we have studied the basics of Reinforcement Learning (RL) including $V(s)$, $Q(s,a)$ and their optimal versions. We have also studied how to estimate them using Monte Carlo and Temporal Difference (TD) methods. We reviewed the two most well-known algorithms Q-learning and SARSA.

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
   - [Distributional Reinforcement Learning](#distributional-reinforcement-learning)
   - [Noisy Nets for Exploration](#noisy-nets-for-exploration)
6. [RAINBOW: Combining Everything We've Learned](#rainbow-combining-everything-weve-learned)
7. [Other Advancements](#other-advancements)
   - [Beyond RAINBOW: The Continuing Evolution](#beyond-rainbow-the-continuing-evolution)
8. [Practical Details](#practical-details)
   - [The Brittleness of Deep Reinforcement Learning](#the-brittleness-of-deep-reinforcement-learning)
   - [The Fundamental Challenge](#the-fundamental-challenge)
9. [Conclusion](#conclusion)


## Deep Learning



Thanks to the advancements in Deep Learning (DL), people started looking at the conjunction between RL and DL. The fundamental question was: if neural networks are so powerful, can they also approximate Q-values? Basically, let the network take as input state representation and output Q-values for all possible actions. And the short answer: they can *but with many hacks*!

--- 

**A quick important note.**

At first glance it might be intuitive to represent this problem as a classification problem. Given state $s$ we want to get an index of the most profitable action $a$, so the network outputs the probability distribution over actions, i.e. let the neural network be our policy. After all we did the same in ES and CEM examples.

However, to use any TD-learning update rules we need to estimate state or state-action values. If the network only outputs the probability distribution over all actions, how can we use that to estimate the Q-values?

Moreover, how can we decide which actions are the best if we do not have any estimates of Q-values? Remember that our rule was to pick the action corresponding to the highest value, so we need an estimate for them. 

Basically, what I am trying to say that it makes much more sense to let the network output not a probability distribution over action space, but do a regression task. Given the state $s$ as the input, the network outputs the estimates of Q-values for each action. As these values are real numbers, this task is inherently of a regression type. 


## Deep Q-Learning

Okay, let's modify a bit our notion of Q-values to $Q(s, a; \theta)$ where $\theta$ are parameters of the network (read it as Q-values obtained from the neural network with parameters $\theta$). We can think of it as a function that takes state $s$ as input and outputs Q-value for all actions. The idea is to train the network to approximate $Q(s, a)$ for all possible actions. 

We cannot just collect the dataset of $(s, [Q(s,a) \  \text{for each} \  a  \in A])$ in the same way as we collected $(s,a)$ pairs in the Cross-Entropy Method, because we simply do not know the true Q-values. So we have to think of a workaround, and you know what, in the previous lesson we already established these workarounds. More specifically, the Temporal Difference or Monte-Carlo learning is exactly what we need. That means that we train the network in the same way we trained vanilla Q-learning!

So, our Q-learning update rule is:

$$
\delta = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t) - Q(s_t, a_t; \theta_t) \\
Q(s_t, a_t; \theta_t) \leftarrow Q(s_t, a_t; \theta_t) + \alpha \delta
$$


But we need to slightly modify it. We are not updating the entries in the table anymore, now we are updating the weights and biases of the neural network that estimates them. So, naturally, we need to define the loss function. We pick a simple, yet effective L2 (MSE) loss:

$$
Q^*(s_t, a_t) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t) \\
L(\theta) = \frac{1}{2} \left( Q^*(s_t, a_t) - Q(s_t, a_t; \theta_t) \right)^2 \\
\theta \leftarrow \theta + \nabla_\theta L
$$

This loss represents an Euclidean distance between two points in the high-dimensional state-action space. First point is our current estimate of Q-value for some $(s, a)$ coordinate. And the second point is the updated estimate using newly generated data sample. By trying to minimize this distance we are moving our estimate towards the new updated estimate. And by doing it iteratively, we hope to converge towards the optimal Q-values.

So the intuition behind Deep Q-Learning is exactly the same as in vanilla Q-learning. We play the episode, at each time step we collect `<s, a, r, s'>` samples. We calculate the new estimate for $Q^*(s,a)$ using newly experienced data. Our updated Q-value is then compared with what the neural network has estimated via L2 loss. Then this loss is backpropagated to update the weights, such that in the future the network would improve its estimate towards the updated version. And this is repeated many times.

So, this is the first approach and you know what? It did not work! I know I know; so many words and it does not work in the end. Let's think why?

## Problems with Deep Q-Learning

Okay, let's start with the first problem that is not that obvious but very serious. Every deep learning problem has one assumption that must be valid. It is called i.i.d. or independent and identically distributed data. Basically saying that the data samples the network is using to learn from are independent from one another. But in our case as we iterate in the environment our data coming back from it is at least temporally correlated, right? Doing action $a_t$ in state $s_t$ will affect the next data sample, because that action itself is the main source of the next state $s_{t+1}$. So, there is a temporal correlation between samples ($s_{t+1}$, $a_{t+1}$) and ($s_t$, $a_t$).

Consider this: imagine doing the same action (*Go Left*) from the initial state (entry to the maze) leading to the left part of the maze several times in a row; these sequences of experiences will be very similar. Training a neural network directly on such consecutive, correlated samples is inefficient. The same will happen if you try to train a neural network to classify images of dogs and cats by first feeding it 1000 images of dogs and then 1000 images of cats. This leads to unstable learning. So, we have to deal with it somehow. 

The second major problem is "chasing own tail". Look at the update rule for our network:

$$
L(\theta) = \frac{1}{2} \left( Q^*(s_t, a_t) - Q(s_t, a_t; \theta_t) \right)^2 \\ 
Q^*(s_t, a_t) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t)
$$

So, our network tries to make the estimate of $Q(s_t, a_t)$ closer to $Q^*(s_t, a_t)$. But look carefully at the equation of $Q^*(s_t, a_t)$, to calculate this value we are using the estimates of the **same** network but of the next state: $\max_{a'} Q(s_{t+1}, a'; \theta_t)$. The neural network is not that all-mighty, changing params to better estimate $Q(s_t, a_t)$ will affect estimation of $Q(s_{t+1}, a^*)$. So, the network is trying to move $Q(s_t, a_t)$ to its target value, but the target value also moves. On the next iteration, it tries again to move its estimate toward another target and this target also moved. 
It is basically like a dog chasing its own tail. 
We did not see this issue when we were studying Q-learning, simply because changing one entry in the table does not affect another entry, they are independent. But when we are using the same network for both estimates, then we are screwed. 

So, we have two problems that require innovation. The first problem was actually easily solved, instead of immediately updating the neural network with newly gathered `<s, a, r, s'>` sample we instead store it in some sort of memory. After enough samples are collected, we are going to sample a random batch of experience from this memory and update the network on this random batch. This way we are not learning from consecutive samples, but from the random samples of our experience. This destroys the temporal correlation and solves our problem. That storing the experience approach was called **Replay Buffer**. In practice it is often implemented as a double-ended queue of some maximum size N to keep only the last N samples. Also, by sampling a batch we are doing mini-batch gradient descent which was shown as an improvement over full-batch gradient descent.

The second problem required a bit more work, but still the solution was simple. The main idea was to make the target estimate $Q(s_{t+1}, a)$ static, but at the same time very close to our best estimate. So, naturally it could be solved by making a separate network that only estimates these next state's Q-values. By doing so we decouple the correlation between $Q(s_{t}, a)$ and $Q(s_{t+1}, a)$ estimates. That's great, right? But how then this separate network learns? How does it improve its estimates of $Q(s_{t+1}, a)$, because if it does not improve then our main network is trying to move towards some incorrect values?

Well, the problem is that we cannot make this separate network learnable, i.e. also update its weight and biases the same way we update the main network's parameters. Because we would come back to our original problem. The smart solution is that once every period (let's say 100 steps) we just copy weights from the main network and that's all! Of course to copy the weights the second network must have exactly the same architecture as the main one, but it is the only condition. 

Think about it, the main network has already updated several times its estimates, so it is way ahead. By copying we make the second network up-to-date with the main one and at the same time we are making the target estimates static! And this is what we originally wanted! Basically, we are replacing one Deep Q-Network with two twins. In the literature the main network is called "online" and the second is called "target". 

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

## Double Deep Q-Network

### Overestimation Problem 
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
y_j = r_{j+1} + \gamma Q(s_{j+1}, \arg\max_{a'} Q(s_{j+1}, a'; \theta_{online}); \theta_{target}) \\
\text{or if next state is terminal} \\
y_j = r_{j+1}  
$$


**Why does this work?** The online network might still pick an action whose value it overestimates. However, **it's much less likely** that the separate, slightly older target network will also happen to overestimate the value of that same specific action. By separating selection and evaluation across the two networks, we break the feedback loop that created the optimism. This leads to more accurate Q-value estimates and, as a result, more stable and better-performing agents.

This small change to the target calculation often provides a significant performance boost and has become a standard technique in modern value-based reinforcement learning.

---

If you remember, in Q-learning, that problem was a motivation for exploring other update rules and coming to SARSA. As we are just reviewing the history of innovations of DQN, I cannot accurately tell you why they have not tried using SARSA or EV-SARSA update rule to try to solve the overestimation problem. What I can do is provide my opinion:

DQN uses a Replay Buffer, which means that the approach is heavily off-policy. The sample that we collect from the Replay Buffer - `<s, a, r, s'>` - may be generated by a very old version of our network and the current network can be several update steps ahead of it.

SARSA, on the other hand, is more on-policy. It evaluates what next action the current policy would have taken, and then uses its Q-value for update. As we had to use a Replay Buffer to break temporal correlation, then it means that we would have needed to store `<s, a, r, s', a'>` samples. Updating our current network (which might be hundreds of update steps ahead) with the sample generated by some older network (which can be hundreds of update steps behind the current network) can be bad. Think of it like you being an adult asking advice from yourself when you were two years old.

Using an up-to-date estimate of Q-value for the next state and the best action is, in my and the authors of DDQN opinion, a much more stable and better approach.


## Prioritized Experience Replay

Okay, the next improvement was not directly affecting DQN, but rather improving the overall training process. We said the batch of experience is sampled randomly from the replay buffer. Choosing randomly breaks the temporal correlation and stabilizes the training, but maybe we can improve it? For example, break temporal correlation and at the same time leverage some prior knowledge about our samples to sample smarter? 

This is what was introduced in Prioritized Experience Replay (PER) paper, the authors showed that picking not at random but with some pre-defined logic results in a better learning curve. 

### What and how do we use for sampling?

Okay, when I told you that we use some pre-defined logic I did not mean that we research the problem and then sample the experience that we think is important. No, I meant that we can leverage something that the agent itself is calculating to improve our sample logic.

Let me be specific, for each tuple of experience `<s, a, r, done, s'>` that we gather there is a corresponding TD error (well, it can be easily computed). This TD error can also serve as a good signal of importance of that sample. Think of it: the bigger the TD error, the more the network was wrong about the estimation, meaning the more the network was surprised. So, naturally, we would want to train on this sample more, to make the surprise much less in future.

This intuition is exactly what PER formalizes. Here's how it generally works:

1. **Assigning Priorities:**

When a new experience (transition) `<s, a, r, done, s'>` is added to the replay buffer, its TD error isn't known yet. So, it's typically given a maximum priority initially to ensure it gets sampled at least once.

When a transition $i$ is sampled from the buffer and used for learning, its TD error $\delta_i$ is calculated: $\delta_i = y_i - Q_{online}(s_i, a_i)$.

The new priority $p_i$ of this transition is then set based on this TD error, usually using its absolute value plus a small positive constant $\epsilon_{PER}$ (to prevent any transition from having zero probability of being sampled): $p_i = |\delta_i| + \epsilon_{PER}$. SO it means that the higher the error (both positive and negative) the bigger the priority of that sample.

To have a more control over the priorities, it is further raised to a power $\alpha$ (a hyperparameter, where $0 \le \alpha \le 1$): $p_i^\alpha$. 
- If $\alpha=0$, we get uniform random sampling, like in a Replay Buffer. 
- If $\alpha=1$, we get full prioritization based on $p_i$.

2. **Sampling Based on Priorities:**

So, instead of sampling uniformly, transitions are now sampled proportionally to their assigned priorities: $p_i^\alpha$. Hence, the probability of sampling transition $i$ becomes: 

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$
where $\sum_k$ means the sum of all transitions' priorities in the memory. This is typically implemented efficiently using a specialized data structure called a SumTree (as seen in the OpenAI Baselines code). A SumTree allows for both efficient updating of priorities and efficient sampling according to these priorities, usually in $O(\log N)$ time, where N is the buffer size.

3. **Importance Sampling (IS) Correction:**

Unfortunately, applying this trick is troublesome. The thing is that sampling transitions non-uniformly like this introduces a bias. We are basically force the network to see "surprising" or high-error samples more frequently than they actually occurred in the agent's interaction with the environment. This can distort the learning process because the distribution of samples used for updates no longer matches the true distribution of experiences. If uncorrected, this can lead the Q-values to converge to incorrect values.

Solution: **Importance Sampling (IS) Weights** 

Importance Sampling is a method used to address a common statistical problem: how to evaluate one strategy using data that was generated by a different one. In our case the first strategy is our true experience of agent interaction with the environment and the second strategy is picking for training only most suprising experience tuples. To being able to safely improve the first strategy using the second we have to re-weight each data point to correct for this mismatch. For each sampled transition $i$, an IS weight $w_i$ is calculated to adjust its contribution to the gradient update. The weight is typically: 
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

This down-weights the updates for transitions that were sampled very frequently (high $P(i)$, low $w_i$) and up-weights those that were sampled rarely (low $P(i)$, high $w_i$), thus correcting for the biased sampling. For example if we sample uniformly then there should be no correction, right? The probability of picking specific tuple $i$ in a uniform distribution is: 

$$
P_i = \frac{1}{N} 
$$
Then given that, the weights are:
$$
w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta = 1
$$

So, basically no correction.

In practice, the weights are usually normalized by dividing by $\max_j w_j$ in the batch to keep update magnitudes stable.

The main benefit of this technique is a significant improvement in sample efficiency. It enables the agent to continuously and safely learn from its entire history in the replay buffer, even as its strategy changes. This ability to effectively reuse past experiences is a key component of many modern reinforcement learning algorithms.

4. **Updating Priorities After Learning:**

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

## Dueling DQN

Okay, you remember how we justified learning Q-values instead of just $V(s)$? That was because knowing $V(s)$ alone isn't enough to pick the best action if we don't have a model of the environment (i.e., we can't easily see what action leads to what next state and its value). With $Q(s,a)$, choosing the best action is easy: just take the $\arg\max_a Q(s,a)$. So, you might wonder, why would anyone want to bring $V(s)$ back into the picture? But it was done in the Dueling DQN paper. 

The insight behind that paper is that for many states, the value of the state itself $V(s)$ is a dominant factor, and the specific action taken might only result in minor differences in outcome. Imagine your agent is in a game state where a missile is inevitably going to hit it, no matter what action it takes. All actions from this state are "bad" because the state itself is terrible. In such cases, learning the Q-value for every single action independently is inefficient. The network would have to learn that $Q(s, \text{duck})$ is very negative, $Q(s, \text{jump})$ is very negative, and $Q(s, \text{run})$ is also very negative, largely because $V(s)$ itself is so negative. 

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

### Why is this better?

- **More Efficient Learning** 

When the environment signals that a state $s$ is good or bad, this information can be learned more directly by the Value Stream. This single update to $V(s)$ then efficiently informs the Q-values of all actions in that state. The network doesn't have to redundantly learn the common state value component for each action's Q-value output.

- **Better Generalization**

The network can learn a good estimate of $V(s)$ even if not all actions have been frequently tried in state $s$. This stable $V(s)$ baseline then helps in evaluating actions more reliably.

- **Improved Performance** 

This factored representation often leads to faster learning and better final policy performance, especially in environments where many states have actions whose values are very similar (i.e., advantages are small).

In conclusion, Dueling DQN doesn't get rid of Q-values; it provides a more intelligent internal structure for the neural network to learn and represent them by decomposition into state values and action advantages. This architectural change can be combined with other improvements like Double DQN and Prioritized Experience Replay.

## N-step return

Another question you might ask is: if we used eligibility traces in tabular Q-learning to propagate rewards back more quickly, can we use them in DQN? The answer is **no**, and it's worth understanding exactly why.

To understand the incompatibility, we first need to recall how eligibility traces work.

### What is an Eligibility Trace?

Think of an eligibility trace as a short-term memory or a "trail of breadcrumbs." When an agent visits a state-action pair $(s, a)$, it leaves a "trace" that marks its visit. This trace then slowly fades away over time. The purpose of this trace is **credit assignment**. When a surprising event happens later on (a high or low TD-error), the credit (or blame) for that surprise is propagated backward along this trail. The more recent a visit, the stronger its trace, and the more credit it receives. This allows a single TD-error to update not just the immediately preceding state-action pair, but the entire recent sequence of pairs that led to it.

### The Core Requirement: A Contiguous Trajectory

For this backward propagation of credit to work, the trace must be **unbroken**. The algorithm needs to know the exact, contiguous sequence of states and actions:

$$
\ldots \rightarrow (s_{t-2}, a_{t-2}) \rightarrow (s_{t-1}, a_{t-1}) \rightarrow (s_t, a_t)
$$

### The Conflict with Experience Replay

The entire purpose of the replay buffer is to destroy this contiguous sequence. It shatters the temporal correlations in the agent's experience. When you sample a mini-batch from the buffer, you get a collection of completely unrelated transitions, for example:

- Transition 1: $(s_{58}, a_{58}, r_{59}, s_{59})$ from episode 3
- Transition 2: $(s_{102}, a_{102}, r_{103}, s_{103})$ from episode 7
- Transition 3: $(s_{23}, a_{23}, r_{24}, s_{24})$ from episode 1

When you calculate the TD-error for Transition 1, you have no way of knowing what state-action pair came before it $(s_{57}, a_{57})$. The "breadcrumb trail" is gone. It's impossible to propagate the error backward along a trace that doesn't exist in the shuffled mini-batch.

Replay buffer helped us break temporal correlation, but it originated another fundamental conflict. This fundamental conflict between the **on-policy** nature of eligibility traces and the **off-policy** nature of experience replay is why they cannot be used together.

---

### The Alternative: N-Step Returns

However, we can achieve a similar goal-looking further into the future to create a more stable learning target—with **N-Step Returns**. Instead of a 1-step target, we can unroll the trajectory for $N$ steps:

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

It is important to understand that we are still storing in the memory each transition of the agent, we are just changing the reward to N-step return. It might be more intuitive to understand this concept with an example. 

Let's say an agent took 5 steps in an environment and our $N$ is set to 3 (for simplicity assume that only last step resulted in terminal next state):

($s_0$, $a_0$, $r_1$, $s_1$) -> ($s_1$, $a_1$, $r_2$, $s_2$) -> ($s_2$, $a_2$, $r_3$, $s_3$) -> ($s_3$, $a_3$, $r_4$, $s_4$) -> ($s_4$, $a_4$, $r_5$, $s_5$, $\text{done}=T$)

Then, we would calculate each 3-step return and these tuples in the form (state, action, return, final state, actual steps taken, done) will be added to our buffer:

- $(s_0, a_0, r_1 + \gamma r_2 + \gamma^2 r_3, s_3, 3, F)$
- $(s_1, a_1, r_2 + \gamma r_3 + \gamma^2 r_4, s_4, 3, F)$
- $(s_2, a_2, r_3 + \gamma r_4 + \gamma^2 r_5, s_5, 3, T)$
- $(s_3, a_3, r_4 + \gamma r_5, s_5, 2, T)$
- $(s_4, a_4, r_5, s_5, 1, T)$

So, we are still adding to the buffer every transition, the only change is replacing one step reward with N-step return. You might wonder why do we have to store additionally actual number of steps taken? This $n$ is needed to correctly calculate TD-target. If you remember for TD-target we use this formula:

$$
\text{TD Target} = r + \gamma \cdot \max_a{Q(s_{t+1},a)}
$$

Since we replaced a single step reward $r$ with N-step return $G_n$, we also have to scale the final state value correspondigly (via $\gamma$). Meaning that the target is calculated now as:

$$
\text{TD Target} = G_n + \gamma^n \cdot \max_a{Q(s_{t+n+1},a)}
$$

## Distributional Reinforcement Learning

Now, here's a fascinating question that probably never crossed your mind: what if we've been thinking about Q-values all wrong? Crazy right? I mean, we've been happily estimating $Q(s,a)$ as a single number - the expected return from taking action $a$ in state $s$. But what if that's not enough? What if we need to understand not just the average outcome, but the entire *distribution* of possible outcomes?

This is exactly the revolutionary insight behind **Distributional Reinforcement Learning**, introduced by Bellemare et al. in their groundbreaking 2017 paper. And trust me, once you understand this concept, you'll never look at Q-values the same way again.

### The Problem with Expectations

Let me give you an intuitive example. Imagine you're an agent deciding between two actions in some game state:

- **Action A**: Gives you exactly 10 points every single time
- **Action B**: Gives you either 0 points (50% chance) or 20 points (50% chance)

Both actions have the same expected value of 10 points. Using traditional Q-learning, your agent would see these actions as equivalent: $Q(s, A) = Q(s, B) = 10$. But are they really the same? Action A is safe and predictable, while Action B is risky but potentially more rewarding. In many scenarios, this distinction matters, depending on the problem you might want to prefer A over B or vice versa. However, when we just represent the actions values as a single real number then this distinction is just erased and we no longer can prefer anything!

The fundamental issue is that **expected values throw away crucial information about uncertainty and risk**. When we collapse the entire distribution of possible returns into a single number, we lose the ability to reason about the variability of outcomes.

### The Distributional Approach

Instead of learning $Q(s,a)$ as a scalar, distributional RL learns $Z(s,a)$ as a **probability distribution over returns**. This distribution captures not just the expected value, but the entire range of possible outcomes and their probabilities.

Mathematically, we define the **return distribution** as:

$$
Z(s,a) = \sum_{t=0}^{\infty} \gamma^t R_{t+1}
$$

where $Z(s,a)$ is now a random variable representing all possible returns, not just their expectation.

The beautiful insight is that we can still recover the traditional Q-value if needed:

$$
Q(s,a) = \mathbb{E}[Z(s,a)]
$$

But now we have access to much richer information about the nature of the returns. That sounds much better, don't you agree?

### The Distributional Bellman Equation

Just as we had a Bellman equation for Q-values, we need one for return distributions. The distributional Bellman equation is:

$$
Z(s,a) \stackrel{d}{=} R(s,a) + \gamma Z(s', \pi(s'))
$$

where $\stackrel{d}{=}$ means "equal in distribution." This equation says that the return distribution from state-action pair $(s,a)$ equals the immediate reward plus the discounted return distribution from the next state.

But here's where it gets tricky: how do we represent and learn these distributions using neural networks? On one hand we can use something like Gaussian Distribution to represent the all possible Q-values for any (state, action) pairs. But that means that the network will have to work with infinite possible returns. Instead it is more computationally efficient to create something like a histogram of Q-values.

### Categorical DQN: Discretizing the Distribution

The first practical solution was **Categorical DQN** (C51), which discretizes the return distribution into a fixed number of "atoms" or bins. Here's how it works:

1. **Choose a Range**: Define the minimum and maximum possible returns, $V_{min}$ and $V_{max}$

2. **Create Atoms**: Divide this range into $N$ discrete support points (typically $N=51$, hence C51):
   $$
   z_i = V_{min} + i \cdot \frac{V_{max} - V_{min}}{N-1}, \quad i = 0, 1, \ldots, N-1
   $$

3. **Learn Probabilities**: For each action, the network outputs a probability distribution over these $N$ atoms. So instead of outputting a single Q-value per action, it outputs $N$ probabilities that sum to 1.

4. **Distributional Target**: When computing the target for the distributional Bellman equation, we need to:
   - Take the target distribution from the next state
   - Shift it by the immediate reward: $r + \gamma z_i$
   - Project it back onto our fixed support (since $r + \gamma z_i$ might not align with our predefined atoms)

### The Projection Step

The projection step is where Categorical DQN shows its ingenuity. When we compute $r + \gamma z_i$, this shifted value rarely lands exactly on one of our predefined atoms. So we need to distribute its probability mass to the nearest atoms.

If $\hat{z}_j = r + \gamma z_j$ falls between atoms $z_l$ and $z_u$, we split the probability proportionally:

$$
\left(\Phi \hat{Z}\right)_l = \frac{z_u - \hat{z}_j}{z_u - z_l} \quad \text{and} \quad \left(\Phi \hat{Z}\right)_u = \frac{\hat{z}_j - z_l}{z_u - z_l}
$$

This ensures that the total probability mass is conserved during the projection.

### Example of the Projection Step

Let's work through a numerical example to see exactly how this projection works:

**Setup:**
- We have $N = 5$ atoms (simplified for clarity)
- $V_{min} = -10$, $V_{max} = 10$
- Our support points are: $z_0 = -10$, $z_1 = -5$, $z_2 = 0$, $z_3 = 5$, $z_4 = 10$
- Current probability distribution: $P = [0.1, 0.2, 0.4, 0.2, 0.1]$
- Immediate reward: $r = 3$
- Discount factor: $\gamma = 0.9$

It means that we assume that the Q-values of that state are in range from [-10, 10] and we are not anymore looking for a single value. We are representing Q-values as a probability distribution of being either [-10, -5, 0, 5, 10]. 

**The Projection Problem:**
When we compute the distributional Bellman update, we need to shift our distribution by $r + \gamma z_i$:

- $r + \gamma z_0 = 3 + 0.9 \times (-10) = 3 - 9 = -6$
- $r + \gamma z_1 = 3 + 0.9 \times (-5) = 3 - 4.5 = -1.5$
- $r + \gamma z_2 = 3 + 0.9 \times 0 = 3$
- $r + \gamma z_3 = 3 + 0.9 \times 5 = 3 + 4.5 = 7.5$
- $r + \gamma z_4 = 3 + 0.9 \times 10 = 3 + 9 = 12$

**The Issue:** These shifted values $\{-6, -1.5, 3, 7.5, 12\}$ don't align with our original support $\{-10, -5, 0, 5, 10\}$!

**The Solution - Projection:**

1. **For $\hat{z}_0 = -6$:** Falls between $z_0 = -10$ and $z_1 = -5$
   - $\left(\Phi \hat{Z}\right)_0 = \frac{-5 - (-6)}{-5 - (-10)} = \frac{1}{5} = 0.2$
   - $\left(\Phi \hat{Z}\right)_1 = \frac{-6 - (-10)}{-5 - (-10)} = \frac{4}{5} = 0.8$
   - Original probability $0.1$ gets split: $0.1 \times 0.2 = 0.02$ to $z_0$, $0.1 \times 0.8 = 0.08$ to $z_1$

2. **For $\hat{z}_1 = -1.5$:** Falls between $z_1 = -5$ and $z_2 = 0$
   - $\left(\Phi \hat{Z}\right)_1 = \frac{0 - (-1.5)}{0 - (-5)} = \frac{1.5}{5} = 0.3$
   - $\left(\Phi \hat{Z}\right)_2 = \frac{-1.5 - (-5)}{0 - (-5)} = \frac{3.5}{5} = 0.7$
   - Original probability $0.2$ gets split: $0.2 \times 0.3 = 0.06$ to $z_1$, $0.2 \times 0.7 = 0.14$ to $z_2$

3. **For $\hat{z}_2 = 3$:** Falls between $z_2 = 0$ and $z_3 = 5$
   - $\left(\Phi \hat{Z}\right)_2 = \frac{5 - 3}{5 - 0} = \frac{2}{5} = 0.4$
   - $\left(\Phi \hat{Z}\right)_3 = \frac{3 - 0}{5 - 0} = \frac{3}{5} = 0.6$
   - Original probability $0.4$ gets split: $0.4 \times 0.4 = 0.16$ to $z_2$, $0.4 \times 0.6 = 0.24$ to $z_3$

4. **For $\hat{z}_3 = 7.5$:** Falls between $z_3 = 5$ and $z_4 = 10$
   - $\left(\Phi \hat{Z}\right)_3 = \frac{10 - 7.5}{10 - 5} = \frac{2.5}{5} = 0.5$
   - $\left(\Phi \hat{Z}\right)_4 = \frac{7.5 - 5}{10 - 5} = \frac{2.5}{5} = 0.5$
   - Original probability $0.2$ gets split: $0.2 \times 0.5 = 0.1$ to $z_3$, $0.2 \times 0.5 = 0.1$ to $z_4$

5. **For $\hat{z}_4 = 12$:** Falls outside our range! We clip it to the boundary $z_4 = 10$
   - All probability $0.1$ goes to $z_4$

**Final Result:**
- $P'_0 = 0.02$
- $P'_1 = 0.08 + 0.06 = 0.14$
- $P'_2 = 0.14 + 0.16 = 0.30$
- $P'_3 = 0.24 + 0.1 = 0.34$
- $P'_4 = 0.1 + 0.1 = 0.20$

**Verification:** $0.02 + 0.14 + 0.30 + 0.34 + 0.20 = 1.00$ ✓

The projection perfectly preserves probability mass while mapping our shifted distribution back onto the fixed support!

### Why Does This Work So Well?

You might wonder: "This seems like a lot of complexity just to get the same expected value in the end. Why bother?"

The answer lies in **richer representations and better learning dynamics**:

1. **Better Value Estimates**: By learning the full distribution, the network captures uncertainty information that helps it make better decisions. States with high variance in returns are naturally distinguished from those with low variance.

2. **Improved Learning Dynamics**: The distributional approach often leads to more stable learning. Instead of a single target that can jump around dramatically, we have a distribution of targets that evolves more smoothly.

3. **Multi-modal Returns**: Some state-action pairs might have genuinely multi-modal return distributions (e.g., "you want to lose small, I want to win big"). Traditional Q-learning would average these modes, potentially missing important structure.

4. **Risk-Sensitive Behavior**: With access to the full distribution, the agent can make risk-aware decisions. It can choose conservative actions when uncertainty is high or aggressive actions when it's confident about high returns.

### Some practical details

Combining Distributional RL with Double DQN or Dueling DQN or N-step is not troublesome, it is pretty straightforward. The only difference is that the network now computes not a scalar but a vector (distribution). Let's review an example of CartPole.

The state consists of 4 values and there are only two actions. Previously, in DQN our network would do something like this: 

1. Accept `[B, 4]` tensor (batched states)
2. Output `[B, 2]` tensor (batches Q-values per each action)

Our logic of picking action was just by picking an index of the highest Q-value of each instance in the batch.

Now, our network output slightly changes:

1. Accept `[B, 4]` tensor (batched states)
2. Output `[B, 2, 51]` tensor (batches probability distribution of Q-values per each action, assuming 51 atoms) 

How can we pick an action if we don't have numbers to compare and pick the highest? The answer is simple: we just take the expected value of each distribution and that is our Q-value. 

---

One more detail that must be discussed: Loss. In DQN we used L2 loss (or Smooth L1 loss), but here we shifted our regression task towards classification task, so do we still use the same loss? No.

We are using the loss that works with probability distributions: **Cross-Entropy loss**. Let's recap the pipeline:

1. We pass a batch of next-states to the online network and get as a result `[B, 2, 51]` tensor
2. We compute the expected value of each distribution to get `[B, 2]` tensor of Q-values
3. We pick the index of the highest Q-value
4. We pass a batch of next-states to the target network and get as a result `[B, 2, 51]` tensor
5. We get `T = [B, 51]` tensor of batched probability distributions of actions that online network thinks are the best
6. We shift this `[B, 51]` by rewards and scale by gamma, since shifted and scaled distribution can be outside of our atoms (bins) we project it.
7. We pass a batch of current-states to the online network and get as a result `[B, 2, 51]` tensor
8. We pick `S = [B, 51]` tensor of batches distribution of the actions that were taken in these states
9. We compute a cross-entropy loss between `S` and `T` as: $-T \cdot log(S)$

---

**The `log_softmax` Trick for Stability**

You might have noticed that in many modern deep learning models, especially for classification, people use `log_softmax` instead of the more intuitive softmax. Our distributional network is essentially a classifier - it's trying to classify the Q-value into one of 51 atomic "bins." So, why bother with the logarithm?

The answer comes down to two key things: numerical stability and efficiency.

- **The Naive Way: softmax then log**

    Let's first look at the straightforward approach, which is what the CleanRL code does.

    - Network Output: Your network outputs raw scores (logits) for each atom: a `[B, 2, 51]` tensor.

    - Get Probabilities: You apply a softmax function to get a valid probability distribution where all values are between 0 and 1 and sum to 1.

    - Calculate Loss: The cross-entropy loss requires you to compute `-(target_dist * log(predicted_dist))`. This means you have to take the `log()` of the probabilities from step 2.

This seems fine, but it hides a nasty numerical trap. If the network is very confident that an atom has zero probability, the softmax output for that atom will be very close to 0. Taking log(0) results in negative infinity (-inf), which makes your loss explode to NaN and kills the training process. This is why the many codes need a small hack to prevent this:

```Python
# The manual "hack" to avoid log(0)
log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5)) 
```

- **The Smart Way: `log_softmax`**

    Now let's look at the more robust method.

    - Network Output: Your network outputs the same raw logits as before.

    - Get Log-Probabilities: Instead of `softmax`, you apply `F.log_softmax` directly to the logits. The network's output is now the logarithm of the probabilities.

    - Calculate Loss: The loss calculation becomes simpler and safer. Since you already have the log-probabilities, you just multiply them by the target distribution: `loss = -(target_dist * predicted_log_dist)`.

The log_softmax function is a single, fused operation that is implemented with a mathematical trick (the Log-Sum-Exp trick). It calculates the final result without ever computing values that could overflow to inf or underflow to log(0). It completely removes the need for the `.clamp()` hack. The only thing to remember is to do `.exp()` to map the log-probabilities to true probabilities.

---

**Calculating TD-error for updating priorities in experience replay**

Right, so previously we were using TD-error (calculated via L2 loss) as our priority for experience replay. How can we calculate them without L2 loss now?

There are two ways to do so:

1. Take expectation of source and target tensor (`S` and `T` from example above) to get Q-values. Subtract one from another, take the absolute value and use it as new priority. This is the simple and yet effictive way.
2. Use the cross-entropy loss calculated above as the measure of new priority.

    The logic is that the cross-entropy loss is the most accurate measure of the "error" or "surprise" for a distributional agent. It doesn't just measure the difference in the average outcome; it measures how different the entire shape of the predicted probability distribution is from the target distribution. A high loss signifies a big surprise, meaning the agent's understanding of that state-action pair was very wrong. Therefore, that experience should have a high priority.

### The Empirical Results

The results were striking. Categorical DQN didn't just match the performance of traditional DQN - it significantly outperformed it across a wide range of Atari games. More importantly, the learned return distributions revealed fascinating insights about the structure of different games. In some games, the distributions were narrow and concentrated (low uncertainty), while in others they were broad and multi-modal (high uncertainty).

This wasn't just a marginal improvement; it was a fundamental shift in how we think about value-based reinforcement learning. By embracing the inherent stochasticity of returns rather than averaging it away, distributional RL opened up new possibilities for more nuanced and effective decision-making.

## Noisy Nets for Exploration

Alright, let's talk about exploration - one of the most persistent challenges in reinforcement learning. You remember $\epsilon$-greedy, right? Pick a random action with probability $\epsilon$, otherwise pick the best action according to your current estimates. It's simple, it works, but let's be honest - it does not sound smart.

Think about it: $\epsilon$-greedy treats all actions equally when exploring. Whether you're in a state where you've tried all actions a thousand times or in a completely novel state you've never seen before, $\epsilon$-greedy just picks randomly. That's like exploring a new city by randomly closing your eyes and walking in arbitrary directions. Sometimes it works, but it's hardly optimal!

### The Deep Problem with $\epsilon$-Greedy in Deep RL

The problem becomes even more pronounced in deep RL. In tabular methods, each state-action pair gets updated independently. But with function approximation, updating one state-action pair affects the values of similar states. This means that traditional exploration strategies might not be leveraging the representational power of neural networks effectively.

Moreover, $\epsilon$-greedy exploration is **temporally inconsistent**. The agent might take action $a$ in state $s$ at time $t$, then immediately take a different action in the same state at time $t+1$, simply due to the randomness in $\epsilon$-greedy. This inconsistency can confuse the learning process and lead to inefficient exploration.

### Enter Noisy Networks

**Noisy Networks** for exploration, introduced by Fortunato et al. in 2017, provide an elegant solution to these problems. Instead of adding noise to the action selection (like $\epsilon$-greedy), noisy networks add noise directly to the neural network parameters. Like we did in Evolution Strategy example, remember? So, the key insight is: **if you add noise to the weights, the entire policy becomes inherently stochastic, but in a structured and consistent way**.

### How Do Noisy Networks Work?

The basic idea is to replace the linear layers in your network with "noisy linear layers." Each noisy layer computes:

$$
y = (\mu^w + \sigma^w \odot \epsilon^w) x + \mu^b + \sigma^b \odot \epsilon^b
$$

Let's break this down into simple terms. Think of the network trying to make a decision.

- **$\mu$ or agent's core strategy** 

    This is the main, learnable part of the network ($\mu^w$ for weights, $\mu^b$ for biases). After training, this represents the agent's best understanding of the optimal policy. You can think of it as the agent's primary, deterministic plan.

- **$\epsilon$ or random idea**

    This is just a vector of random noise that we sample from a standard distribution. It provides a random direction to deviate from the main plan. Think of it as a random "what if we tried this instead?" suggestion that gets injected into the system.

- **$\sigma$ or exploration scale** 

    This is the most brilliant part. This sigma is a set of parameters that the network learns. It acts like a volume knob for the random epsilon vector. The agent learns to turn this knob up or down based on its experience.

So, the logic is as follows:

- If the agent is in a state where it's uncertain, it can learn to increase $\sigma$, amplifying the noise and encouraging it to explore different actions.

- If the agent is in a familiar state where it's very confident, it can learn to decrease $\sigma$ towards zero, effectively silencing the noise and sticking to its main plan $\mu$.

The final weight used in the calculation is essentially: (Core Strategy + Confidence Knob * Random Idea).

The truly powerful insight here is that the network isn't just learning the optimal policy $\mu$; it's also learning how and when to explore by tuning the noise scale $\sigma$. This allows for much more intelligent, state-dependent exploration than the simple, random approach of epsilon-greedy.

### Gradient flow

Okay, it is probably a point of confusion. How does the neural network knows which states are well-known and which are not, how does it know when to decrease exploration scale and when to increase. Let's see the gradient flow. 

During training, we want to minimize a loss function $L$ (e.g., the Mean Squared Bellman Error in DQN). The gradients of the loss with respect to the learnable parameters ($\mu$ and $\sigma$) are calculated via backpropagation.

Let's focus on the gradient for a single weight $W_{ji}$ (connecting input neuron $i$ to output neuron $j$) and its corresponding bias $b_j$. The gradient of the loss $L$ with respect to any parameter $\theta$ is found using the chain rule:

$$\frac{\partial L}{\partial \theta} = \sum_k \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial \theta}$$

Let's denote the incoming gradient from the next layer (or the loss function), $\frac{\partial L}{\partial y_j}$, as $g_j$. We can now find the partial derivatives of the output $y_j$ with respect to each of the four learnable parameters associated with it.

The output of a single neuron $j$ is:

$$y_j = \sum_i W_{ji} x_i + b_j = \sum_i (\mu_{ji}^w + \sigma_{ji}^w \cdot \epsilon_{ji}^w) x_i + (\mu_j^b + \sigma_j^b \cdot \epsilon_j^b)$$

#### 1. Gradient with respect to the weight mean, $\mu_{ji}^w$:

To find $\frac{\partial y_j}{\partial \mu_{ji}^w}$, we treat all other parameters as constants.

$$\frac{\partial y_j}{\partial \mu_{ji}^w} = x_i$$

Therefore, the full loss gradient for the weight mean is:

$$\frac{\partial L}{\partial \mu_{ji}^w} = g_j \cdot x_i$$

#### 2. Gradient with respect to the weight standard deviation, $\sigma_{ji}^w$:

To find $\frac{\partial y_j}{\partial \sigma_{ji}^w}$, we again treat other parameters as constants.

$$\frac{\partial y_j}{\partial \sigma_{ji}^w} = \epsilon_{ji}^w \cdot x_i$$

Therefore, the full loss gradient for the weight standard deviation is:

$$\frac{\partial L}{\partial \sigma_{ji}^w} = g_j \cdot (\epsilon_{ji}^w \cdot x_i)$$

#### 3. Gradient with respect to the bias mean, $\mu_j^b$:

$$\frac{\partial y_j}{\partial \mu_j^b} = 1$$

Therefore, the full loss gradient for the bias mean is:

$$\frac{\partial L}{\partial \mu_j^b} = g_j$$

#### 4. Gradient with respect to the bias standard deviation, $\sigma_j^b$:

$$\frac{\partial y_j}{\partial \sigma_j^b} = \epsilon_j^b$$

Therefore, the full loss gradient for the bias standard deviation is:

$$\frac{\partial L}{\partial \sigma_j^b} = g_j \cdot \epsilon_j^b$$

### How the Updates Differ

By comparing the resulting gradients, the mathematical distinction becomes clear:

### Update for $\mu$ (Core Strategy)
The gradient $\frac{\partial L}{\partial \mu_{ji}^w} = g_j \cdot x_i$ is identical to the gradient for a standard deterministic weight in a linear layer. It updates the baseline, deterministic behavior of the network based on the input activation $x_i$ and the error signal $g_j$.

### Update for $\sigma$ (Exploration Scale)
The gradient $\frac{\partial L}{\partial \sigma_{ji}^w} = g_j \cdot \epsilon_{ji}^w \cdot x_i$ contains the additional term $\epsilon_{ji}^w$. This term is the specific random number that was sampled and used for that weight in that forward pass.

This means the update to $\sigma$ is directly modulated by the random perturbation that was just attempted.

- If a specific perturbation $\epsilon_{ji}^w$ led to a better-than-expected outcome (i.e., the sign of $g_j$ aligns with the sign of $\epsilon_{ji}^w$ to produce a positive update for $\sigma$), the gradient will increase the value of $\sigma_{ji}^w$, encouraging more exploration for that weight.

- Conversely, if the perturbation led to a worse outcome, the gradient will push $\sigma_{ji}^w$ towards zero, suppressing future exploration for that weight.

The gradient descent algorithm does not need to "know" which parameter was responsible; the mathematical structure of the partial derivatives naturally assigns the updates according to each parameter's distinct role in the forward pass computation.

### Noise: Independent vs Factorized

There are two ways to sample the noise:

1. **Independent Gaussian**: Each weight gets its own independent noise sample. This gives maximum flexibility but requires more random number generation.

2. **Factorized Noise**: This is the method used in the original paper and is much more efficient. Instead of generating noise for every single connection, we generate just two small random vectors: one for the input neurons and one for the output neurons. We then combine them to create the full noise matrix.

Let's make that concrete. Imagine a layer with 3 input neurons and 2 output neurons. The full noise matrix needs to be [2, 3].

- The Slow Way (Independent): We would generate 2 * 3 = 6 random numbers to fill the matrix.

- The Clever Way (Factorized):

We create a random vector for the inputs, of size 3: `noise_in = [ε_in1, ε_in2, ε_in3]`

We create a random vector for the outputs, of size 2: `noise_out = [ε_out1, ε_out2]`

We combine them with an outer product. This just means we multiply each element of `noise_out` by the entire `noise_in` vector to create the final [2, 3] matrix:

```
[ ε_out1 * ε_in1,  ε_out1 * ε_in2,  ε_out1 * ε_in3 ]
[ ε_out2 * ε_in1,  ε_out2 * ε_in2,  ε_out2 * ε_in3 ]
```

The benefit is huge. We only had to generate 3 + 2 = 5 random numbers instead of 6. For a big 512x512 layer, that's the difference between generating 1,024 numbers versus over 262,000. This shortcut dramatically reduces the computational cost while still producing a rich, structured noise that works exceptionally well for exploration. It's the standard, go-to approach for Noisy Nets.

### Why This is Better

1. **State-dependent Exploration**: The network learns to explore more in uncertain states and less in well-understood states. This is because the noise affects the entire forward pass, and the learned noise parameters can adapt to different situations.

2. **Temporal Consistency**: Unlike $\epsilon$-greedy, if you put the agent in the same state multiple times with the same noise realization, it will take the same action. This consistency helps with learning.

3. **Automatic Annealing**: As the network becomes more confident (through training), it can learn to reduce the noise scales $\sigma$, naturally transitioning from exploration to exploitation without manual scheduling.

4. **Action Diversity**: Instead of uniform random selection, the noisy network provides structured exploration that's informed by the current value estimates.

### The Learning Dynamics

Here's what makes noisy networks particularly clever: **the noise parameters are updated using the same gradients as the mean parameters**. When the agent discovers a good action through noisy exploration, the gradient update not only reinforces the mean parameters toward that action but also adjusts the noise parameters appropriately.

If exploration in a particular direction led to good outcomes, the network might learn to increase noise in that direction. If random exploration consistently leads to poor outcomes in certain states, the network learns to reduce noise for those states.

### Practical details

Integrating Noisy Nets into a modern DQN agent like Rainbow requires careful attention to a few key details, especially concerning how the online and target networks interact.

First, the most obvious step is replacing the standard `nn.Linear` layers in the final parts of your network (typically after the convolutional base) with your new `NoisyLinear` layers. This is what injects the trainable noise into the decision-making process.

However, a critical danger with Noisy Nets is the "certainty trap." An agent can quickly become confident in a suboptimal policy, leading to predictably low rewards. Because the outcomes are no longer surprising, the TD error drops to near zero. The network interprets this as high certainty and learns to decrease its noise parameter $\sigma$, effectively halting exploration and trapping the agent. To mitigate this and ensure a stable start, a common practice is to implement a warm-up period. For the first N thousand steps (a parameter often called learning_starts), the agent takes purely random actions to populate its replay buffer. This provides a diverse foundation of experience, preventing the agent from immediately converging to a poor local optimum before it has had a chance to see a broader range of possibilities.

But more importantly to review these two critical implementation question: 
- **If the noise is for exploration, should our target network also be noisy?**
- **When we select the best action for the next state, we use the online network: `argmax Q_online(s', a')`. Should this action selection be noisy?**

Here, the answer is contraversial. The noise in the online network represents the agent's current exploratory policy. Therefore, we should use the noisy online network to select the action, as this reflects the agent's current belief about the best action under its exploration strategy.

On the other hand, the main Q-learning update rule is to assign to current Q-value the immediate reward plus the discounted optimal value of the next state. The optimal value represents the biggest Q-value. Exploration noise can make the online network to pick the wrong action which can lead to non-optimal next state value. So, in this sense, we should not use noisy nets. 

Same with the target network. The entire purpose of the target network is to provide a stable, low-variance anchor for our Bellman updates. Adding noise to the target would re-introduce the very instability we are trying to prevent. 

My logic and intuition says that we should disable noise when we are calculating target for TD-error. However, the de-facto official implementation of [RAINBOW](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py) is not doing that. So, I don't have a concrete answer for you. I guess it depends, you should experiment yourself to see the difference.

---

Another important detail that might seem strange at first: after each learning update, we must reset the noise in our network. Why do we do this? If the noise is what drives exploration, why would we constantly change it?

The answer is that resetting the noise is what enables temporally-consistent, state-dependent exploration.

Let's break down the logic:

- A Fixed "Personality": When we call `reset_noise()`, we sample a new set of random noise vectors ($\epsilon_w$ and $\epsilon_b$) for every noisy layer. These noise vectors are then fixed until the next reset. This means that for a short period (between one learning update and the next), the agent's policy is completely deterministic. If you put it in the same state twice during this period, it will take the exact same action. This gives the agent a consistent "personality" or "quirk" for its next few interactions with the world.

- Coherent Exploration: This consistency is vital. It allows the agent to explore coherently. Instead of just flailing randomly from one step to the next like ϵ-greedy, the agent can follow a consistent (but noisy) strategy for a sequence of steps, which is a much more effective way to discover the consequences of a particular behavior.

- Preventing Stagnation: If we never reset the noise, the agent would have one fixed "quirky" personality for its entire lifetime. It would be stuck with its initial random set of preferences and would not truly explore. By resetting the noise after each gradient update, we give the agent a new "personality" for the next round of data collection.


## RAINBOW: Combining Everything We've Learned

Now we arrive at the crown jewel of the DQN family: **RAINBOW DQN**. But this isn't just another incremental improvement—it's a masterpiece of algorithmic engineering that demonstrates how individual innovations can be combined synergistically.

The name "RAINBOW" isn't just catchy marketing; it represents the integration of six different colors (improvements) of the DQN spectrum:

1. **Double DQN** - Addressing overestimation bias
2. **Prioritized Experience Replay** - Learning from important experiences
3. **Dueling DQN** - Separating state values from action advantages
4. **Multi-step Learning** - N-step returns for better targets
5. **Distributional RL** - Learning return distributions instead of expectations
6. **Noisy Networks** - Parameter space exploration instead of $\epsilon$-greedy

### The Integration Challenge

You might think: "Just throw all these techniques together and call it a day!" But combining these improvements is far from trivial. Each technique was originally developed and tested in isolation, and their interactions can be complex and sometimes counterproductive.

For example:
- How do you prioritize experiences when you're learning distributions instead of scalar values?
- How do noisy networks interact with the target network copying mechanism?
- Does the dueling architecture work well with distributional outputs?

The RAINBOW authors didn't just combine these techniques—they carefully engineered how they interact, resolving conflicts and ensuring that each component enhances rather than interferes with the others.

### The Distributional-Dueling Integration

One particularly elegant integration is between distributional RL and dueling networks. In standard dueling DQN, we compute:

$$
Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)
$$

But in RAINBOW, we need to do this for entire distributions, not just scalar values. So the distributional dueling architecture computes:

$$
Z(s,a) = V_Z(s) + \left( A_Z(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A_Z(s,a') \right)
$$

where each component ($V_Z$, $A_Z$) is now a distribution over the support atoms. This requires careful handling of the probability distributions at each step.

### Multi-step Distributional Targets

Similarly, combining N-step returns with distributional RL requires rethinking the target computation. Instead of the scalar N-step target:

$$
G_n = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a')
$$

RAINBOW uses a distributional N-step target where the final term becomes a distribution:

$$
G_n = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n Z(s_{t+n}, a^*_{t+n})
$$

where $a^*_{t+n}$ is selected using the expected values of the distributions (maintaining the double DQN action selection/evaluation separation).

### The Remarkable Results

The results were nothing short of spectacular. RAINBOW didn't just incrementally improve upon existing methods - it achieved a massive leap in performance across the Atari benchmark. More impressively, it demonstrated that these improvements were truly synergistic. The full RAINBOW agent significantly outperformed agents using any subset of its components.

Perhaps most importantly, RAINBOW established a new standard for what deep RL algorithms could achieve. It showed that careful engineering and systematic combination of well-understood improvements could lead to dramatic advances in performance.

### The Legacy of Systematic Improvement

RAINBOW represents something profound about the field of deep reinforcement learning: **progress often comes not from revolutionary new ideas, but from the careful integration of incremental improvements**. Each component of RAINBOW addressed a specific, well-understood limitation:

- Overestimation bias → Double DQN
- Sample efficiency → Prioritized Experience Replay
- Architectural efficiency → Dueling Networks
- Credit assignment → Multi-step returns
- Information loss → Distributional RL
- Exploration efficiency → Noisy Networks

By systematically addressing each limitation and carefully engineering their interactions, RAINBOW achieved performance that was greater than the sum of its parts. This approach—identifying specific problems, developing targeted solutions, and carefully combining them—became a template for future algorithm development in deep RL.

The success of RAINBOW also demonstrated the maturity of the DQN paradigm. It showed that value-based deep RL had evolved from a promising but unstable technique to a robust and powerful framework capable of achieving superhuman performance across a wide range of challenging domains.


## Other Advancements

Even beyond RAINBOW, the field continues to evolve with improvements like:
- **Implicit Quantile Networks (IQN)** for better distributional learning
- **NGU (Never Give Up)** for improved exploration in sparse reward environments
- **Agent57** which achieved superhuman performance on all 57 Atari games
- Various forms of **model-based enhancements** that combine planning with value-based learning

The key insight here is that deep RL research follows a pattern of **incremental, targeted improvements**. Each advancement identifies a specific limitation (bias, variance, exploration, representation, etc.) and proposes a focused solution. Understanding the foundational improvements we've covered gives you the tools to understand and appreciate these more advanced techniques as they continue to emerge.

## Practical details

### The Brittleness of Deep Reinforcement Learning

We must confront an uncomfortable truth about deep RL: **it is notoriously brittle and sensitive to hyperparameter choices**. Unlike supervised learning where you might get "pretty good" results with default parameters, deep RL can completely fail with seemingly minor parameter changes. This brittleness is not a bug, it's a fundamental characteristic of the field that stems from several factors:

1. **Non-stationarity**: The data distribution changes as the policy improves, making the learning problem inherently unstable
2. **Bootstrapping**: We're learning from our own predictions, which can amplify errors
3. **Exploration-exploitation trade-offs**: Wrong exploration parameters can lead to never discovering good policies
4. **Credit assignment**: Rewards are often sparse and delayed, making it hard to know what actions were truly responsible for success

This means that **hyperparameter tuning is often the hardest and most time-consuming part of deep RL research**. What works for one environment might completely fail for another, and small changes can mean the difference between superhuman performance and complete failure. 

The hardest truth about deep RL is that **there is no universal set of hyperparameters**. What works for Atari will likely fail on continuous control. What works for dense rewards will fail with sparse rewards. This is why hyperparameter tuning often consumes 80% of research time in deep RL projects. For example, I can tune hyperparameters to let DQN solve Cartpole environment, then I can deploy same set of hyperparameters to let DDQN also try. However, it can achieve worse results, despite being an improved version of DQN. 

The field is actively working on solutions - from automatic hyperparameter tuning to more robust algorithms - but for now, success in deep RL requires patience, systematic experimentation, and a deep understanding of how each parameter affects the learning dynamics.

This brittleness isn't a flaw to be embarrassed about, it's a reminder that we're working at the frontier of what's possible in machine learning, where the interactions between algorithms, environments, and implementations create a complex landscape that we're still learning to navigate.

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

### **Distributional Reinforcement Learning: Beyond Expectations**
Perhaps the most revolutionary conceptual breakthrough came with **Distributional RL**, which challenged the fundamental assumption that we should learn expected values. By learning entire return distributions instead of just their expectations, agents gained access to crucial information about uncertainty and risk. **Categorical DQN (C51)** demonstrated that this richer representation leads to more stable learning and better performance, while providing insights into the inherent variability of different state-action pairs.

### **Noisy Networks: Intelligent Exploration**
Moving beyond the crude randomness of $\epsilon$-greedy exploration, **Noisy Networks** introduced parameter space exploration that is both structured and learnable. By adding trainable noise to network parameters, agents learn not just what actions to take, but how and when to explore. This approach provides temporally consistent, state-dependent exploration that automatically balances exploration and exploitation as learning progresses.

### **RAINBOW: The Synergistic Masterpiece**
The culmination of these individual advances came with **RAINBOW DQN**, which demonstrated that careful integration of multiple improvements could achieve performance greater than the sum of its parts. RAINBOW's success established a template for systematic algorithmic development in deep RL, showing how targeted solutions to specific problems can be combined synergistically.

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