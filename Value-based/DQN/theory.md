# Deep Reinforcement Learning

Well, we have studied the basics of Reinforcement Learning (RL) including $V(s)$, $Q(s,a)$ and their optimal versions. We have also studied how to estimate them using Monte Carlo and Temporal Difference (TD) methods. We reviewed the two most well-known algorithms Q-learning and SARSA. But you probably noticed that the problems that we solved were quite simple. But you heard about RL playing video-games at human-level and even beating humans at them. So, what's the problem?

The Q-learning and SARSA algorithms are working well with small problems, by small I mean that the problem has finite and more importantly small state-action space. We can store all the pair in a table and look at each row and column to get the Q-value. But as the problem get bigger the state-space gets bigger as well. We conclude to the point that no computer in the world can hold that big table. So, we need to find a way how we can approximate Q-value without storing it somewhere.

## Deep learning

Thanks to the advancements in Deep Learning people started looking at conjuction between RL and DL. The foundamental question was: if neural networks are so powerful, can they approximate Q-values? Let the network take as input state representation and output Q-values for all possible actions. And short answer: they can (but with many hacks)!

At first glance it might be intuitive to represent this problem as classification problem, given state $s$ we want to get an index of the most profitable action $a$. 
But then we would not be able to use any TD-learning rules, because if the network estimates best actions, how can we compute Q-values? Moreover, who can decide which actions is the best if we don't have any estimates of Q-values? Remember that our rule was to pick action corresponding to the highest Q-value, so it makes much more sense to estimate Q-values instead of directly outputting index of the best action. So, this problem is inherently a regression problem. 

## Deep Q-learning

Okay, let's denote Q-values as $Q(s, a; \theta)$ where $\theta$ are parameters of the network (read it as Q-values obtained from the neural network with parameters $\theta$). We can think of it as a function that takes state $s$ as input and outputs Q-value for all actions. The idea is to train the network to approximate $Q(s, a)$ for all possible actions.

We train it in the same way we trained Q-learning using TD-learning! 

So, our TD-learning update rule is:

$$
\delta = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t) - Q(s_t, a_t; \theta_t) \\
Q(s_t, a_t; \theta_t) \leftarrow Q(s_t, a_t; \theta_t) + \alpha \delta
$$

But now we need to re-design this rule, because we are not updating the Q-values, but we update the weight and biases of the network that estimates these values. So, naturally, we need to define the loss function. We pick a simple, yet effective L2 (MSE) loss 
$$
L(\theta) = \frac{1}{2} \left( Q*(s_t, a_t) - Q(s_t, a_t; \theta_t) \right)^2 \\ 
Q*(s_t, a_t) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta_t)
$$

So, we try to minimize the error between our estimation and the TD-target that we compute, this way the network will be able to better and better estimate the Q-values. 

And this was the first approach and it failed! I know, so many words and it does not work in the end. Let's think why?

## Problems with Deep Q-learning

Okay, let's start with first problem that is not that obvious but very serious. Every deep learning problem has one assumption that must be valid. It is called i.i.d. or independent and identically distributed data. Basically saying that the data is sampled independently, but in our case as we iterate with environment our data coming back from it is at least temporarily correlated, right?  Imagine doing the same action from the initial state leading to the left part of the maze several times in a row; these sequences of experiences will be very similar. Training a neural network directly on such consecutive, correlated samples is inefficient and can lead to unstable learning or a network that overfits to recent experiences, forgetting past ones. So, we have to deal with it somehow. 

Another major problem is the "chasing own tail". Looking back at our loss function, we estimate Q-values for the current state and for the next state using the same network. Unfortunately, the neural network is not that all-mighty, changing params to better estimate $Q(s_t, a)$ will affect estimation of $Q(s_{t+1}, a)$. It makes the network to basically chase its own tail. We did not have this problem when we were using tables, because changing one entry in the table did not affect another, but wehn we are using the same network for both estimates we are wrong. 

So, we have two problems that require innovation. First problem was actually easily solved, people introduced a Replay Buffer. It is data collection that stores experience (in the form of $<s, a, r, s`>$) and then we sample from it randomly. This way we are not learning from consecutive samples, but from the random samples of our experience, which destroys the temporal correlation. In practice it is implemented as FIFO collection from which we sample a batch of data, by improving the overall gradient descent, as in traditional supervised learning. 

Second problem required a bit more work, but still the solution is simple. Instead of having one network for two estimates, we will have two networks. One network estimates $Q(s_t, a)$ and the second network estimates $Q(s_{t+1}, a)$. This way we will stabilize the learning and solve this problem. In practice, the two networks are called target and online networks, having exactly same architecture. We are updating only weight of the online network and do not optimize target network. But with time we periodically just copy weights from online network to the target network, to keep it up to date with our training. 

And that's it, this two innovations resulted in a beatiful paper that was able to learn how to play variety of Atari games just from pixels! There was no pre-defined policy of how to act when someone is shooting us, the agent learnt how to interpret the pixels into Q-values to make the best move! 

## Deep Q-Network (DQN)

So, the pseudo-code for DQN is

1. Initialize target and online networks
2. Copy params from online network into target network: $\theta_{target} \leftarrow \theta_{online}$
3. Initialize a replay buffer $D$ with some size $L$
4. for episode $e$ do 

   4.1 Choose action $a_t$ given our $\epsilon$-greedy policy

   4.2 Apply action $a$ and obtain $r, s_{t+1}, d$ where $d$ is a boolean to check if the episode is terminated

   4.3 Store $<s_t, a_t, r, d, s_{t+1}>$ in $D$

   4.4 Sample $B$ batch size from $D$ 

     4.4.1 Compute TD-target $y_j = r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta_{target})$ (set $y_j = r_j$ if the step was terminal) for all $j \in B$ 

     4.4.2 Compute loss and apply gradient descent


And that is able to solve complex problems! 

## Advancements of DQN

This paper was introduced in 2013, so naturally people started to improve it. Here I will mention some I think that are most important.

### Overestimation Problem and Double DQN
The two-network scheme (online and target networks) solved the problem of "chasing its own tail" by providing a more stable target for the updates. However, another subtle issue remained, particularly with the way the target Q-value was calculated: the overestimation bias.

Let's look at the target calculation in the original DQN again:

$y_j = r_{j+1} + \gamma \max_{a'} Q(s_{j+1}, a'; \theta_{target})$

The $\max_{a'}$ operator is the culprit here. When the Q-value estimates from the target network ($Q(s_{j+1}, a'; \theta_{target})$) are noisy or uncertain (which they often are, especially early in training), the max operator will tend to pick the action $a'$ whose Q-value is overestimated due to this noise. It preferentially selects positive estimation errors. Using this potentially inflated maximum Q-value in the target y_j means the online network Q(s_j, a_j; \theta_{online}) is consistently updated towards an overly optimistic target. This can lead to the learned Q-values being systematically higher than the true optimal Q-values, potentially leading to suboptimal policies. 😬

Double Deep Q-Network (DDQN) to the Rescue! 💡

Double DQN, introduced by Hado van Hasselt et al., provides a simple yet effective fix to this overestimation problem. The core idea is to decouple the action selection from the action evaluation when forming the target.

Instead of using the target network for both selecting the best next action and evaluating its Q-value, DDQN does this:

Action Selection: Use the online network $\theta_{online}$ to determine the action $a^*_{online}$ that it thinks is best in the next state $s_{j+1}$: $a^*_{online} = \arg\max_{a'} Q(s_{j+1}, a'; \theta_{online})$

Action Evaluation: Then, use the target network $\theta_{target}$ to get the Q-value of that specific action $a^*_{online}$: $Q(s_{j+1}, a^*_{online}; \theta_{target})$

So, the DDQN target $y_j$ becomes:

$y_j = r_{j+1} + \gamma Q(s_{j+1}, \arg\max_{a'} Q(s_{j+1}, a'; \theta_{online}); \theta_{target})$

(Set $y_j = r_{j+1}$ if $s_{j+1}$ is terminal).

Why does this work? The online network might still pick an action whose value it overestimates. However, it's less likely that the target network (with its different, slightly older parameters) will also overestimate the value of that same specific action. By separating selection and evaluation across the two networks, DDQN often leads to more accurate Q-value estimates and more stable, better-performing agents. The rest of the DQN algorithm (replay buffer, periodic target network updates via copying $\theta_{online} \rightarrow \theta_{target}$) remains the same.

This small change often provides a significant performance boost!

### Sampling smart from Replay Buffer

Okay, next iteration was not directly improving DQN, but improving the training process. We desribed previously that we sample randomly from replay buffer, it is unbiased approach, because we basically choose randomly. But maybe we can leverage some prior knowledge before sampling? 

This is what was introduced in Prioritized Experience Replay (PER) paper, they showed that picking not at random but with some logic results in better learning curve. 

What and how do we use for samping? 

Well, for each tuple of experience $<s,a,r,d,s`>$ we can compute the TD error. Intuitively it can be understood as how the network was suprised by this experience, the bigger the error the more it was suprised. So, we want to train on this sample more, to make the network better. 

This intuition is exactly what PER formalizes. Here's how it generally works:

1. Assigning Priorities:

When a new experience (transition) $(s, a, r, s', \text{done})$ is added to the replay buffer, its TD error isn't known yet. So, it's typically given a maximum priority initially to ensure it gets sampled at least once.

When a transition i is sampled from the buffer and used for learning, its TD error $\delta_i$ is calculated (as we did in the DQN update: $\delta_i = y_i - Q_{online}(s_i, a_i)$).

The new priority $p_i$ of this transition is then set based on this TD error, usually using its absolute value plus a small positive constant $\epsilon_{PER}$ (to prevent any transition from having zero probability of being sampled): $p_i = |\delta_i| + \epsilon_{PER}$

Sometimes, this priority is further raised to a power $\alpha$ (a hyperparameter, where $0 \le \alpha \le 1$) to control the degree of prioritization: $p_i^\alpha$. If $\alpha=0$, we get uniform random sampling. If $\alpha=1$, we get full prioritization based on $p_i$.

2. Sampling Based on Priorities:

Instead of sampling uniformly, transitions are now sampled proportionally to their assigned priorities (or $p_i^\alpha$). The probability of sampling transition $i$ becomes: $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$

This means transitions with higher priorities (larger TD errors) are more likely to be selected for the training mini-batch.

This is typically implemented efficiently using a specialized data structure called a SumTree (as seen in the OpenAI Baselines code). A SumTree allows for both efficient updating of priorities and efficient sampling according to these priorities, usually in $O(\log N)$ time, where N is the buffer size.

3. The Bias Problem and Importance Sampling (IS) Correction:

Unfortunately, this trick has its own pitfalls. Sampling transitions non-uniformly like this introduces a bias. The network sees "surprising" or high-error samples more frequently than they actually occurred in the agent's interaction with the environment. This can distort the learning process because the distribution of samples used for updates no longer matches the true distribution of experiences. If uncorrected, this can lead the Q-values to converge to incorrect values.

The Solution: **Importance Sampling (IS) Weights** 

To counteract this bias, PER uses Importance Sampling weights. For each sampled transition i, an IS weight $w_i$ is calculated to adjust its contribution to the gradient update. The weight is typically: $w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta$

N is the current size of the replay buffer.
P(i) is the probability with which transition i was sampled (calculated above).
$\beta$ is another hyperparameter ($0 \le \beta \le 1$) that controls how much correction is applied. It is usually annealed (linearly increased) from an initial value (e.g., 0.4) towards 1 over the course of training. Starting with a smaller $\beta$ helps stabilize learning early on when Q-estimates are very noisy.

Using the Weights: These weights $w_i$ are then used to scale the loss for each transition in the mini-batch. For example, if the loss for sample i is $L_i = \delta_i^2$, the weighted loss becomes $w_i \cdot \delta_i^2$. 

The gradient update for that sample is effectively scaled by $w_i$. $\text{Weighted Loss} = \frac{1}{\text{BatchSize}} \sum_i w_i \cdot \delta_i^2$ 

This down-weights the updates for transitions that were sampled very frequently (high $P(i)$, low $w_i$) and up-weights those that were sampled rarely (low $P(i)$, high $w_i$), thus correcting for the biased sampling. The weights are often normalized by dividing by $\max_j w_j$ in the batch to keep update magnitudes stable.

4. Updating Priorities After Learning:

Once a transition i from the mini-batch has been used for a learning update and its new TD error $\delta_i$ is known, its priority $p_i$ in the buffer must be updated to reflect this new error. This ensures that future sampling correctly prioritizes it based on its current "surprise" level.

In summary, PER changes the DQN training loop like this:

Store new transitions in the buffer with high initial priority.

When sampling a batch: 

a. Sample proportionally to $p_i^\alpha$. b. Calculate IS weights $w_i$ using the current $\beta$.

When computing the loss for the batch, scale each sample's loss by its IS weight $w_i$.

After the update, update the priorities of the sampled transitions in the buffer using their new absolute TD errors.

Anneal $\beta$ towards 1.

By focusing on "surprising" transitions while correcting for the induced bias, PER often leads to significantly faster learning and better final performance compared to uniform sampling. It's a clever way to make the learning process more efficient!

### Learning State Value and Action Advantage (Dueling DQN) 🧠⚔️

Okay, you remember how we justified learning Q-values instead of just $V(s)$? That was because knowing $V(s)$ alone isn't enough to pick the best action if we don't have a model of the environment (i.e., we can't easily see what action leads to what next state and its value). With $Q(s,a)$, choosing the best action is easy: just take the $\arg\max_a Q(s,a)$. So, you might wonder, why would we want to bring $V(s)$ back into the picture with Dueling DQN?

The insight behind Dueling DQN is that for many states, the value of the state itself ($V(s)$) is a dominant factor, and the specific action taken might only result in minor differences in outcome. Imagine your agent is in a game state where a missile is inevitably going to hit it, no matter what action it takes. All actions from this state are "bad" because the state itself is terrible. In such cases, learning the Q-value for every single action independently might be inefficient. The network would have to learn that $Q(s, \text{duck})$ is very negative, $Q(s, \text{jump})$ is very negative, and $Q(s, \text{run})$ is also very negative, largely because $V(s)$ itself is so negative.

Dueling DQN proposes that it's more efficient to separately estimate the state value $V(s)$ and the advantage $A(s,a)$ for each action.

State Value $V(s)$: Represents how good or bad it is to be in state $s$ in general.

Advantage $A(s,a)$: Represents how much better or worse taking action $a$ is compared to the other actions possible in state $s$. It's defined as: $A(s,a) = Q(s,a) - V(s)$

The network architecture is modified to have two separate streams (or "heads") after some initial shared layers:

One stream (the Value Stream) outputs a single scalar value, which is an estimate of $V(s)$.

The other stream (the Advantage Stream) outputs a vector of values, one for each action, estimating $A(s,a)$ for each action.

Combining Value and Advantage:

We can then reconstruct the Q-values using the relationship $Q(s,a) = V(s) + A(s,a)$. However, to ensure that V(s) truly learns the state value and A(s,a) learns the advantages in a stable and unique way (avoiding the "identifiability problem"), a specific aggregation layer is used:

$Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)$

By subtracting the mean of the advantages, this formulation forces the advantages for a state to sum to zero. This helps $V(s)$ become a more robust estimate of the actual state value, and $A(s,a)$ represents the true relative preference for each action.

Why is this better?

More Efficient Learning: When the environment signals that a state s is good or bad, this information can be learned more directly by the Value Stream. This single update to V(s) then efficiently informs the Q-values of all actions in that state. The network doesn't have to redundantly learn the common state value component for each action's Q-value output.

Better Generalization: The network can learn a good estimate of V(s) even if not all actions have been frequently tried in state s. This stable V(s) baseline then helps in evaluating actions more reliably.

Improved Performance: This factored representation often leads to faster learning and better final policy performance, especially in environments where many states have actions whose values are very similar (i.e., advantages are small).

So, Dueling DQN doesn't get rid of Q-values; it provides a more intelligent internal structure for the neural network to learn and represent them by decomposing them into state values and action advantages. This architectural change can be combined with other improvements like Double DQN and Prioritized Experience Replay.

### N-step return

Well, this is something that we already seen in Q-learning. Instead of using 1-step return, we can make n-more steps and obtain n-step return. It was shown that this trick improves training! Of course to make it work you need to slightly re-work the replay buffer or PER to contain not a single reward, but a list of rewards (or compute return directly) and instead of $s_{t+1}$ you need to store a $s_{t+n}$