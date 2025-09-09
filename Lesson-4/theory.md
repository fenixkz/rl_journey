# Introduction to RL. Part 4. Policy Gradient.

We have studied a significant portion of Reinforcement Learning (RL) where we focused on Value-Based Learning. This name originates from the fact that our learning process primarily revolved around estimating the action-value function $Q(s,a)$. Using these estimates, we could then derive a policy (typically a greedy or $\epsilon$-greedy policy), where for any given state $s$, we would choose the action corresponding to the highest $Q(s,a)$ value. So, although we were implicitly arriving at a optimal policy, we weren't learning the policy parameters directly. This chapter of RL explores how we can use policies and more importantly how we can improve them.

One key advantage of directly learning a policy is that it can learn truly stochastic policies, which can be optimal in certain situations.

# Theory

Let's recall that our policy $\pi(a | s)$ is a probability distribution over all possible actions $a$ that agent can perform given that the agent is in state $s$. How are we able to learn it?

First of all, in policy-based methods, our policy $\pi_\theta(a|s)$ is typically represented by a function approximator, like a neural network, parameterized by $\theta$ (same as in CEM, ES, or DQN). This network takes the state $s$ as input and outputs a probability distribution over all actions (for discrete actions). Then, we sample from this distribution to choose an action, making the whole action selection process stochastic.

This contrasts with standard DQN. While DQN uses an $\epsilon$-greedy policy for exploration during training (which is stochastic), its ultimate learned optimal policy is deterministic: once Q-values converge, it always chooses the action with the highest Q-value from the same state. Policy Gradient (PG) methods, however, can learn inherently stochastic optimal policies.

## Stochastic vs Deterministic

Let us build some intuition about why stochastic policies can sometimes be better or worse.

It's easy to see cases where stochastic policies might seem problematic. Imagine you are driving towards a tree; the action we must take is to turn to avoid the crash. If our policy is stochastic, due to random sampling, it might still choose the action that corresponds to moving straight, which would be disastrous! (Though a well-trained stochastic policy would assign very low probability to such an action, but it is never zero).

On the other hand, there are cases when using a deterministic policy is a clear disadvantage. Imagine playing Rock-Paper-Scissors. Playing a deterministic policy means making the same move every time (or a predictable sequence). This is a bad idea because your opponent can easily adapt and win. So, playing a somewhat random (stochastic) strategy can achieve better results in the long run against an adaptable opponent.

Another significant advantage of directly learning a stochastic policy is that it naturally incorporates exploration. In DQN, we had to construct schemes like $\epsilon$-greedy to explicitly choose random actions with some probability $\epsilon$. With a stochastic policy $\pi_\theta(a|s)$, the agent inherently tries different actions based on their learned probabilities, providing a more natural way to explore. Something like we saw in Noisy Net in the previous lecture.

Okay, let's now see how we work directly with policies.

## Policy Gradient

As you probably remember, to optimize something, we usually define an objective function. In deep learning for supervised tasks, this is often a loss function that we aim to minimize. In policy optimization, like we already seen before, our objective is the expected total reward, which we want to maximize.

### Objective Function

Our goal is to find the parameters $\theta$ of a policy $\pi_\theta(a|s)$ that maximize the expected total discounted reward. Let:

- $\pi_\theta(a|s)$ is an agent's policy parametrized by $\theta$, i.e. a neural network
- $\tau$ be a trajectory: $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T)$. This is simply a sequence of steps of the agent in an environment.
- $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}$ be the total discounted reward for the trajectory $\tau$.
- $p(\tau; \theta)$ be the probability of observing trajectory $\tau$ when following the policy $\pi_\theta$.


Then, the objective function we want to maximize is the expected return:

$$
J(\theta) = E_{\tau \sim \pi_\theta} [R(\tau)]
$$

This can also be written as a sum over all possible trajectories weighted by their probabilities:

$$
J(\theta) = \sum_{\tau} p(\tau; \theta) R(\tau)
$$

The thing is that there can be incountable amount of all possible trajectories that the agent can end up with, but we are only interested in trajectories that result in the most expected return.
Thus, the main idea of policy-based learning is to adjust $\theta$ (parameters) of our policy such that trajectories with high $R(\tau)$ are more probable.

To maximize $J(\theta)$, we perform gradient ascent on the policy parameters $\theta$:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

where $\alpha$ is a learning rate. The core task now is to find a usable expression for $\nabla_\theta J(\theta)$.

### Deriving $\nabla_\theta J(\theta)$

Let's start by taking the gradient of the objective function:

$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} p(\tau; \theta) R(\tau)
$$

We can interchange the gradient and summation:

$$
= \sum_{\tau} \nabla_\theta p(\tau; \theta) R(\tau)
$$


The term $\nabla_\theta p(\tau; \theta)$ represents a probability of ending up with this specific trajectory using our policy. But it is tricky to work with directly, as $p(\tau; \theta)$ is a complex product of probabilities.  However, we can use a clever identity known as the log-derivative trick (or likelihood ratio trick). Recall that for any differentiable positive function $f(x)$, its derivative $\nabla_x f(x)$ can be rewritten using its logarithm: $\nabla_x \log f(x) = \frac{1}{f(x)} \nabla_x f(x)$. Rearranging this gives $\nabla_x f(x) = f(x) \nabla_x \log f(x)$.

Applying this to our case (with $f(x) = p(\tau; \theta)$ and $x = \theta$):

$$
\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)
$$

Substitute this back into our gradient expression:

$$
\nabla_\theta J(\theta) = \sum_{\tau} p(\tau; \theta) \nabla_\theta \log p(\tau; \theta) R(\tau)
$$

This sum can now be rewritten as an expectation under the policy $\pi_\theta$:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} [ \nabla_\theta \log p(\tau; \theta) R(\tau) ] \tag{1}
$$


Okay, you might think, "What the France! You just said that $p(\tau; \theta)$ is hard to work with, but you just put it under the log, it is still there!" Yes, but stay with me, the magic happens next.

The probability of a specific trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1})$ given our policy $\pi_\theta$ is:

$$
p(\tau; \theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)
$$

Or in other words, it's the probability of the trajectory starting in some specific initial state times the cumulative product of the probabilities of each action we took (given the state we were in) and the probabilities of the environment transitioning to the next state (given our state and action).

And the only thing that we know or at least can compute is $\pi_\theta(a_t | s_t)$ (basically the output of our neural network). I wish there was a way to get rid of these other terms...

Luckily, the logarithm has a wonderful property: $\log(X \cdot Y) = \log X + \log Y$. Applying this:

$$
\log p(\tau; \theta) = \log p(s_0) + \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t | s_t) + \log p(s_{t+1} | s_t, a_t) \right)
$$


Now, here's the key insight: $p(s_{t+1} | s_t, a_t)$ is the environment's transition probability, and $p(s_0)$ is the distribution of initial states. Neither of these depends on our policy parameters $\theta$. They are characteristics of the environment itself. Thus, when we take the gradient with respect to $\theta$, these terms vanish:

- $\nabla_\theta \log p(s_0) = 0$

- $\nabla_\theta \log p(s_{t+1} | s_t, a_t) = 0$

This leaves us with a much simpler expression:

$$\nabla_\theta \log p(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)$$

In other words, the gradient of the log-probability of a trajectory (with respect to our policy parameters) is simply the sum of the gradients of the log-probabilities of the actions taken in that trajectory! Incredible, right? Don't tell me wishes never come true again!

These $\log \pi_\theta(a_t|s_t)$ terms are things our policy network computes (or are directly related to its outputs), and we can calculate their gradients. And it is very easy to compute given the modern auto-gradident libraries, like PyTorch or TensorFlow. The policy (neural network) outputs a probability distribution over all possible actions, you just pick the specific probability of the specific action, compute the logarithm of it and backpropogate to obtain the gradients.

Substitute this expression for $\nabla_\theta \log p(\tau; \theta)$ back into Equation 1:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) R(\tau) \right] \tag{2}
$$

This is a widely known form of the **Policy Gradient Theorem**.

We can even improve this further. We know that the action $a_t$ taken at time $t$ can only influence rewards from $r_{t+1}$ onwards; it cannot influence past rewards $r_1, \dots, r_t$. Therefore, when considering the impact of $\nabla_\theta \log \pi_\theta(a_t | s_t)$, it's more appropriate (and can reduce the variance of the gradient estimate) to multiply it by the return obtained from that step $t$ onwards, rather than the total return $R(\tau)$ for the whole trajectory.

Let $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ be the discounted return starting from time step $t$ until the end of the episode/trajectory.

The policy gradient can then be written as:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right] \tag{3}
$$

This form is the basis of our update rule. We want to adjust our policy parameters $\theta$ in the direction of this gradient:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

So, basically, what PG theorem states:

> Run a trajectory using current policy. At each step compute the derivative of the logarithm of the probability of taken action given the state, multiply it be the total return after it. Sum all of it together. And repeat it for many different trajectories, weight each sum by the probability of that trajectory. And you will get a true direction to improve your current policy towards the optimal one.

Clearly, number of possible trajectories can be practically infinite, so we can't compute the true direction. But what we can do is to estimate it. We can run one trajectory and move the policy in direction of $\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$. Clearly that is not the original "true" direction, but only a noisy estimate of it. But repeating the same procedure again and again will lead thep policy in the correct direction.

## Building the intuition

Now, let's take a minute to discuss and give ourselves some room for thinking. Policy gradient is one of the most important concepts in modern reinforcement learning and we have to have a deeper understanding of the matter.

### Digesting $\nabla_\theta \log \pi_\theta(a_t | s_t)$

First of all, it is important to understand this term - $\nabla_\theta \log \pi_\theta(a_t | s_t)$ - because it is the main tool that actually change the policy. Since $\pi_\theta(a_t | s_t)$ is always positive - because it is the probability - then the logarithm of it is a monotonically increasing function. That means that at any point of this function the gradient of $\log \pi_\theta(a_t | s_t)$ always points towards increasing $\pi_\theta(a_t | s_t)$.

In simpler words, this term represents a vector in the parameter (weight and biases) space and it is **always** points in the exact direction of making $a_t$ more likely to be chosen in state $s_t$. To make the parameter change in the opposite direction, i.e. make $a_t$ less likely to be chosen in $s_t$, the only option is to reverse the direction and it can be done only by multiplying it by a negative number.


Intuitively, this update rule can be understood as follows:
- We sample an action, $a_t$, given the current state, $s_t$, from $\pi_\theta(a|s_t)$
- If this action resulted in a good subsequent return, then it means that $G_t$ is positive. So, by applying a gradient ascent in direction of $\log \pi_\theta(a_t | s_t)$ we are increasing the likelihood of taking that action $a_t$ in state $s_t$ again.
- Conversely, if the action was bad, it resulted in negative $G_t$. We want to move in the opposite direction of $\log \pi_\theta(a_t | s_t)$ to decrease its probability and that can be done only if we multiply by the negative return.

You might wonder then what happens in the environments where the return cannot be negative? Like in CartPole any return is positive, how then this theorem works?

Well, this vanilla PG theorem can still be used to solve CartPole and here is why. Even if there is no built-in mechanism of explicitly decreasing the probability of bad actions it is done implicitly. By increasing the probability of some good action in some state we implicitly decreasing the probability of other actions, because the total sum of a probability distribution has to be always one.

### Objective vs Loss

Another subtle point is what loss should we use to update neural networks?

Our true goal is to maximize the total discounted reward, $J$, over a trajectory. We've initialized a neural network that processes a state and returns a policy (probability distribution over all possible actions). This policy is used to select actions, step in the environment, and collect rewards.

But if we impose this question to ourselfs:

> Our neural network doesn't directly compute our true objective function, $J$. All it does is tell us what actions to choose. So how does optimizing the network also maximize $J$?

There is an implicit connection, of course. By choosing better actions, we collect better rewards, thus maximizing $J$. But what about the math? How does the gradient "know" about this implicit connection and which way to go?

The main trick in all PG algorithms is to choose a clever loss function for the neural network. This loss function acts as a surrogate or a proxy for our true objective, $J$. We don't need to compute $J$ itself, we just need a computable function whose gradient points us in the right direction to improve $J$.

For anyone who has tackled a classification problem, the Negative Log-Likelihood (NLL or Cross-Entropy) loss is a familiar tool. In short, it's a metric that compares the model's predicted probabilities to the actual outcomes. A lower loss means the predictions are closer to the ground truth. You might be surprised to learn, however, that PG methods avoid this loss entirely, despite the problem being a natural fit for classification.

---

**A Simple Analogy: The Shifted Parabola**

Imagine our true objective is to find the minimum of the function $g(x) = x^2 + z(y)$. Let's pretend that the $z(y)$ part is very difficult to calculate, so we can't work with $g(x)$ directly.

However, we have access to a much simpler, surrogate function: $f(x) = x^2$.

Let's look at their gradients (their derivatives):

The gradient of our true objective is
$\frac{d}{dx}g(x) = 2x$.

The gradient of our surrogate is
$\frac{d}{dx}f(x) = 2x$.

They have the exact same gradient!

This means that if we take a step to minimize our easy-to-calculate function $f(x)$, we are also taking a step in the correct direction to minimize our true, hard-to-calculate objective $g(x)$. The gradients tell us the direction of steepest ascent, and since they are identical, the path to the minimum is the same for both functions, even though their actual values are different.

---

**Connecting the Analogy to Policy Gradient**

This is precisely the trick we use. Our true, hard-to-calculate objective is $J$. And the loss we choose to minimize should be a proxy or a surrogate function to our true objective. Our candidate:

$$
\text{Loss} = -G_t \log \pi_\theta(a_t | s_t)
$$

The PG theorem proves that the gradient of this loss is exactly the negative of the gradient of our true objective function!

$$
\nabla_\theta \text{Loss} = -G_t \nabla_\theta \log \pi_\theta(a_t | s_t) = -\nabla_\theta J
$$

Now, think about what our optimizer does. It performs gradient descent:

$$
\text{new}_\theta = \text{old}_\theta - \text{learning\_rate} \times \nabla \text{Loss}
$$

If we substitute $\nabla \text{Loss}$ with $-\nabla J$, we get:
$$
\text{new}_\theta = \text{old}_\theta - \text{learning\_rate} \times (-\nabla J)
$$
$$
\text{new}_\theta = \text{old}_\theta + \text{learning\_rate} \times \nabla J
$$

This is the formula for gradient ascent!

So, by minimizing our cleverly chosen surrogate loss, we are automatically performing gradient ascent on the true objective, $J$. And that is how the neural network maximizes the overall return without even computing it!

# REINFORCE algorithm

Policy-Gradient update rule (Equation 3) gave birth to one of the earliest and most fundamental policy-gradient algorithms, called REINFORCE (also known as Monte Carlo Policy Gradient).

The REINFORCE algorithm approximates the expectation $E_{\tau \sim \pi_\theta}[\cdot]$ by sampling trajectories using the current policy $\pi_\theta$. Here's a conceptual outline:

1. Initialize the policy network parameters $\theta$ (e.g., weights and biases of a neural network).
2. Loop for a number of training iterations:
    1. Get initial state from the environment - $s_t$
    2. Run one full episode using the current policy $\pi_\theta$
        1. The policy $\pi_\theta(a|s_t)$ outputs action probabilities (or parameters for an action distribution).
        2. Sample an action $a_t$ from this distribution.
        3. Execute action $a_t$ in the environment, observe the next state $s_{t+1}$, reward $r_{t+1}$, and whether $s_{t+1}$ is terminal.
        4. Store the tuple $(s_t, a_t, r_{t+1}, \text{done}_t)$ and $\log \pi_\theta(a_t|s_t)$ in some temporal memory.
    3. Calculate discounted return for current episode: $G_t = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{T-t-1} r_T$
        1. Iterate backwards from the last step $T-1$ to the first step $t=0$ for all rewards in the memory.
        2. For each $r_t$, calculate the discounted return: $G_t = r_{t+1} + \gamma G_{t+1}$, starting with $G_T = 0$.
    4. Compute policy gradient estimate
        1. For each corresponding $G_t$ and $\log \pi_\theta(a_t | s_t)$ calculate: $G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$.
        2. Sum these terms up across all steps: $L = -\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$.
    5. Apply a gradient descent step: $\theta \leftarrow \theta - \alpha \hat{L}$

REINFORCE is powerful because it directly optimizes the policy to maximize rewards, but it can have high variance in its gradient estimates due to its reliance on full Monte Carlo returns $G_t$. This high variance can make learning slow or unstable, which motivated later developments like Actor-Critic methods.

## Practical details

There is often a big gap between the elegant theory in a research paper and its practical implementation in code. To bridge this, researchers have continually experimented with techniques to improve training stability and convergence speed. Let's begin our discussion with one of the most fundamental of these improvements: normalization.

### Normalization

The REINFORCE update rule is

$$
\theta \leftarrow \theta + \alpha \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

The return term can be very different depending on the problem. As a rule of thumb, neural networks do not really like extreme values. For example in CartPole the reward is +1 for every step the pole is balanced. Thus, the maximum possible return (assuming for simplicity $\gamma \approx 1$) is $G_t = 500$. It means that the gradients of the neural network would be scaled by 500 leading to a huge update step that can be de-stabilize the training process. The de-facto standard is to use Gaussian-normalized values (zero mean and unit variance). Therefore, normalizing returns (within one episode) helps to make the learning process more stable.

Moreover, by squashing all returns to a zero mean and unit variance, we make some returns positive and some negative. As a result it also helps to solve our problem of not being able to penalize bad actions in the environments with no negative returns.

However, despite all the advantages sometimes it can backfire. You might have even noticed this in an environment like CartPole! Let's dig into why this "good practice" can harm.

**Loss of Absolute Performance Signal**

When we normalize the returns $G_t$ within a single episode (by subtracting the mean of that episode's $G_t$'s and dividing by their standard deviation), we are essentially ranking actions based on how well they performed relative to other actions in that specific episode.

- Imagine one episode where the agent does terribly and only lasts 15 steps. All $G_t$ values will be low. Normalization will still scale these so some look "good" (above the terrible mean) and some look "bad" (below the terrible mean). The agent might reinforce actions that were just "less terrible."
- Imagine one episode where the agent can balance the pole for 100 steps. The actions taken towards the end has a smaller return than the actions taken in the beginning. But does it mean that they are always worse? Of course no. Action taken at step 70 might be very good, it just happened that action taken at step 90 was so bad that the agent lost. Normalization will make $a_{70}$ look bad, because it is just happened that its return is lower than the average return, and PG will make it less likely.

The main problem is, the agent might lose a clear signal about what constitutes an absolutely good outcome (like an episode of 200 steps) versus an absolutely bad one (15 steps). If all normalized returns end up in a similar range (e.g., roughly -1 to +1), it can be harder for the agent to strongly differentiate between an action that led to a truly long episode versus one that led to a mediocre one.

So, usually normalization helps to stabilize the updates for a neural network, but you should be cautious about it.

### Entropy Exploration

There's another piece we often add to the loss function for the policy network called **entropy**. Why do we need it? We need it because it improves exploration.

I know that we already defined that stochastic policies are better than deterministic ones because they inherently push the agent to explore more. But as the agent becomes very confident in its actions, then the stochastic policy becomes more and more deterministic, and we want to keep some order of exploration. So, we need a way to encourage our agent to keep a bit of "curiosity" or "randomness," (especially early in training), so it does not settle down too quickly. In DQN, we had $\epsilon$-greedy exploration, which was a bit... blunt. For policy gradients, we can be more elegant by using the concept of entropy.

**What is Entropy?**

You've probably heard of entropy, maybe from a physics class or information theory. People often describe it as a measure of "chaos" or "disorder," which is a good starting point. In our world of RL, it’s even more helpful to think of entropy as a **measure of randomness**.

Imagine our policy, $\pi_\theta(s,a)$, is about to pick an action.
- If the policy is totally unsure what to do (e.g., for 3 actions, the probabilities are [0.33, 0.33, 0.33]), then we have no idea what's coming next. The outcome is very random! This is a **high-entropy policy**.
- If the policy is almost certain what it's going to do (e.g., [0.98, 0.01, 0.01]), we're not surprised at all when it picks that main action. The outcome is predictable. This is a **low-entropy policy**.

This idea is captured neatly in the classic formula. Let's quickly break it down intuitively:

$$
H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

---

Entropy plays a crucial part later when we discuss Soft Actor-Critic. But since we introduced it here, it might be helpful to understand the roots of this term.

The formula above was introduced by Claude Shannon, a brilliant mathematician, when he tried to solve a very practical problem: how to reliably transmit messages over a noisy channel (like a telegraph wire). To be able to solve it, he needed a way to measure the "amount of information" in a message.

His main idea was that the amount of information in a message is proportional to how surprising it is:

- A message that says "The sun will rise tomorrow" contains almost no information because it's not surprising. Everybody knows that the sun will rise tomorrow.

- A message telling you the result of a fair coin flip contains some information. There is a 50/50 chance of the message being either tail or head.

- A message telling you the winning lottery numbers contains a huge amount of information because it's an extremely surprising outcome.

Entropy was his mathematical definition for the average amount of surprise or uncertainty in a source of information. Shannon was investigating further, he then decided that a measure of information (or "surprisal," let's call it $\mathcal{I}$) should have one crucial property: additivity. If you have two independent events (like flipping two separate coins), the total information you get from learning both outcomes should be the sum of the information from each one.

- Information(Coin 1 AND Coin 2) = Information(Coin 1) + Information(Coin 2)

However, we know that the probabilities of two independent events multiply:

- Probability(Coin 1 AND Coin 2) = Probability(Coin 1) * Probability(Coin 2)

So, Shannon needed a function for $\mathcal{I}$ that turns multiplication into addition. The logarithm is the unique function that does exactly this: $\log(x \cdot y) = \log(x) + \log(y)$. This is why the information content, or surprisal, of an event with probability $p$ is defined as $\log(1/p)$, which is equivalent to $-\log(p)$:

$$
\mathcal{I} = -log(P)
$$

- High Probability (e.g., p=0.99): $-\log(0.99) \approx 0.01$. Very little surprise.
- Low Probability (e.g., p=0.01): $-\log(0.01) \approx 4.6$. A lot of surprise.
- Certainty (p=1): $-\log(1) = 0$. Zero surprise, zero information.

And entropy is simply the expected value (or average) of the surprise. To get the average surprise of a policy, you take the surprise of each possible action and weight it by the probability of that action actually occurring, which results in the formula above.

---

The formula explicitly says: a policy that is surprising all the time (very random) will have a high average surprise, and thus, high entropy.

**So, why is this useful for RL?**

It all comes down to the fundamental trade-off between exploitation (using the best strategy you've found so far) and exploration (trying new things to find something even better). Adding the entropy term to our objective function gives the agent a secondary goal. We are effectively telling it:

> Your main job is to maximize rewards. However, I'll also give you a small bonus for keeping your options open and staying curious.

This "curiosity bonus" discourages the policy from becoming too confident and deterministic too quickly. It pushes the agent to keep exploring, making it less likely to get stuck with a "good enough" strategy and more likely to find the truly optimal one.

People familiar with machine learning can think of this term as a regularization penalty. The regulartization helps the model not to overfit, which in our context means to be extremely confident in the action-choosing.

#### Adding Entropy to the Objective Function

So, how do we actually use this? We add the expected entropy of our policy to our main objective function, which the agent is trying to maximize. Our new objective becomes:

$$
J_{\text{total}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \right] + \beta \cdot \mathbb{E}_{s \sim d_\pi, a \sim \pi_\theta(\cdot|s)} \left[ H(\pi_\theta(\cdot|s_t)) \right]
$$

The agent's part of the loss function (which we are typically minimizing, so we use negative signs for things we want to maximize) looks something like this:

$$
L_{\text{actor}}(\theta) = -\sum_t \left[ G_t \log \pi_\theta(a_t|s_t) - \beta \sum_a \pi_\theta(a|s) \log \pi_\theta(a|s) \right]
$$

Here:

- $\beta$ (beta) is a small positive coefficient (a hyperparameter you tune) that controls the "strength" of the entropy bonus.

When we take the gradient of this new objective with respect to $\theta$ and perform gradient ascent (or gradient descent on the negative objective), the entropy term adds an extra push. This "entropy gradient" encourages changes to $\theta$ that increase the entropy of the policy - that is, make the action probabilities more uniform.

Think of $\beta$ as a knob in exploration/exploitation trade-off. Finding the right balance for $\beta$ (often through experimentation, or by scheduling it to decrease over time) is key. Many advanced algorithms like A3C and PPO almost always include this entropy bonus in their objective functions because it consistently leads to more robust and effective learning. It's a simple trick that makes a big difference!

# Actor Critic

Okay, so we've seen how to estimate the policy gradient using Monte Carlo returns. But as you remember we had the same battle in Value-based approach about waiting for full trajectories or updating at each time step. The REINFORCE algorithm has a high variance, because each trajectory is different from another (we basically improve it each time, so of course it will differ). But what we can do with this variance?

One of the steps that we can take is to reduce the variance in $G_t$ (well it is the main source of high variance). $G_t$ is the estimate of how good the action $a_t$ was: how much (discounted) rewards  were gained after this action was taken. Given the stochastic nature of the policy and the environment, performing same action $a_t$ in the same state $s_t$ can result in very different $G_t$. Imagine, by the random chance the agent has won after taking this action and $G_t$ is high, or it can easily be the other way around agent can sample a different action $a_{t+1}$ or can transit to a different state $s_{t+1}$ (because of stochastic env), and result in low $G_t$. This is the source of high variance. **How can we reduce these fluctuations?**

Solution is to introduce some baseline $b(s_t)$ and subtract it from the return $G_t$. Let's $b(s_t)$ be the baseline for $s_t$, it will represent what is the expected (average) return from this state. Then, by sutracting it from our estimate of return, we can smooth the noise in random fluctuations.

You should probably remember that we already coined a term for that expression: "the expected (average) return the agent obtains if it starts from this state and then acts according to $\pi_\theta$"? Exactly, our baseline is basically $V^{\pi}(s_t)$. It means that our solution to reduce the variance in policy-gradient method is to inject value-based approach!

This is the main idea of Actor-Critic method. The name is coming from the two main components of this method:
- Actor (policy) that acts, i.e. choosing action $a_t$ given state $s_t$
- Critic (value) that critics, i.e. saying how higher or lower total return that action obtained in comparison to the average return from this state

So, we converged to the point of combining both worlds!

## Why does baseline help and why it does not affect the overall objective function

With the baseline, the policy gradient becomes:


$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

However, the baseline function is only a function of the state and is independent of the action (it is basically a constant), so subtracting it doesn’t affect the expected value of the policy gradient. That is, the expected policy gradient remains unchanged:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] \\
= E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right] - E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

To make the objective function unchanged, the second term must be equal to zero:

$$
 E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] \\
 = \sum_{t=0}^{T-1} E_{\tau \sim \pi_\theta} \left [ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

Now, let's focus on a single term in this sum for a specific time step $t$. The expectation $E_{\tau \sim \pi_\theta}$ is over trajectories, which means it involves an expectation over states $s_t$ that can occur at time $t$ and actions $a_t$ taken in those states according to $\pi_\theta$.

For a specific step $t$, we can write the expectation as:
$$
E_{s_t \sim p_t(\cdot;\theta)} \left[ E_{a_t \sim \pi_\theta(\cdot|s_t)} \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] \right]
$$
where $p_t(s_t; \theta)$ is the probability distribution of states at time $t$ under policy $\pi_\theta$ (imagine you collected billion of transitions using policy $\pi_\theta$ and $p_t(s_t; \theta)$ would be a number of times this state $s_t$ present in this data).

Let's expand the inner expectation (the one over actions $a_t$ for a given $s_t$):
$$
E_{a_t \sim \pi_\theta(\cdot|s_t)} \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] = \sum_{a} \pi_\theta(a_t | s_t) \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$
Since $b(s_t)$ does not depend on $a_t$ (it's a function of state $s_t$ only), we can pull it out of this inner summation:
$$
= b(s_t) \sum_{a} \pi_\theta(a_t | s_t) \nabla_\theta \log \pi_\theta(a_t | s_t)
$$
Now, remember the log-derivative trick we used earlier: $\nabla_\theta \pi_\theta(a_t | s_t) = \pi_\theta(a_t | s_t) \nabla_\theta \log \pi_\theta(a_t | s_t)$.

So, the sum becomes:
$$
= b(s_t) \sum_{a} \nabla_\theta \pi_\theta(a_t | s_t)
$$

We can swap the summation and the gradient operator:
$$
= b(s_t) \nabla_\theta \sum_{a} \pi_\theta(a_t | s_t)
$$

And here's the crucial step. For any state $s_t$, the sum of probabilities of taking all possible actions $a_t$ according to the policy $\pi_\theta(a_t | s_t)$ must be equal to 1 (because the policy is a probability distribution):
$$
\sum_{a} \pi_\theta(a_t | s_t) = 1
$$

Substituting this back:

$$
= b(s_t) \nabla_\theta (1)
$$

The gradient of a constant with respect to $\theta$ is 0:

$$
= b(s_t) \cdot 0 = 0
$$

Therefore, for any given state $s_t$ at time t:

$$
E_{a_t \sim \pi_\theta(\cdot|s_t)} \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] = 0
$$

Okay, so the objective function is unaffected by introducing the baseline, but why does it help in reducing the variance?

- Without changing the expected policy gradient, the role of the baseline is to serve as a reference value, used to reduce the background "noise" in the reward that does not depend on the current action selection. By subtracting an appropriate baseline, the scores for different actions become more focused and stable, which reduces variance.
- Intuitively: If no baseline is subtracted, the reward received after taking an action may vary significantly due to the randomness of the environment. Subtracting a suitable baseline makes the relative quality of different actions clearer, reducing randomness and lowering the variance of the policy gradient.

## How to compute $b(s_t)$

Okay, so the baseline $b(s_t)$ helps in reducing the variance of our policy gradient estimates. The most common choice for this baseline is an estimate of the state-value function $V^\pi(s_t)$ for the current policy $\pi$. And now it's time to thank our existing knowledge in Value-based RL, because we indeed have methods to estimate these state-values!

In actor-critic methods, the component responsible for learning this baseline $V^\pi(s_t)$ is called the critic. It is typically a function approximator (like a neural network) with its own set of parameters, let's say $\phi$. So, the critic learns an estimate

$$V_\phi(s_t) \approx V^\pi(s_t)$$

How does it learn $V_\phi(s_t)$? As you guessed correctly, using Temporal Difference (TD) Learning!

The actor takes a step in the environment, resulting in a transition $(s_t, a_t, r_{t+1}, s_{t+1})$. The critic observes this transition and updates its value estimate $V_\phi(s_t)$, using the TD target:
$$
y_t = r_{t+1} + \gamma V_\phi(s_{t+1}) \\
\text{or if next state is terminal} \\
y_t = r_{t+1}
$$

Given the target, we can calculate the TD error to update the critic's parameters:
$$
\delta_t = y_t - V_\phi(s_t) = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Exactly like in Deep Q-Network, we define a loss (L2 loss or smooth L1 loss):

$$
L(\phi) = \delta_t^2 = (r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2
$$

And we use gradient descent to update $\phi$ with a separate learning rate $\alpha_C$:
$$
\phi \leftarrow \phi - \alpha_C \nabla_\phi L(\phi)
$$


This is exactly like what we have already seen in previous lesson! The only difference is that now we estimate only one real number, $V(s)$, and not $Q(s,a)$ for all possible actions.

---

Wait a minute. In the previous lesson, we established that a direct TD approach for Q-learning is unstable. The two problems we identified - temporal correlation and a moving target - caused this direct approach to fail. So, how can it work for the critic in Actor-Critic? The short answer is: these problems still exist, but they are less severe. This leads to the question: why can't we apply the same solutions we derived for DQN?

Let's start with the target network to solve the moving target problem. First, we have to understand why the moving target problem is less severe in on-policy Actor-Critic methods. In DQN we were trying to learn the **optimal** Q-values ($Q^*$), i.e. the best possible Q-values for every state-action pair. Given the problem, most probably that those optimal Q-values are unique and constaint. So, ultimately we were trying to learn the true, static and unique distribution of optimal Q-values.

On the other hand, in AAC, the critic's job is fundamentally different. It's not trying to learn the universal, optimal state-value function. Instead, its goal is to learn the value of the actor's current policy, $V^\pi$. The state-values differ from one policy to another on constrast to their optimal versions. Because the actor's policy, $\pi$, is constantly improving, the true value function for that policy, $V^\pi$, is also expected to change. What I am trying to say is: the target is supposed to move. This makes the problem less severe because the actor and critic are a team, moving and improving together.

But nonetheless, there is actually a debate whether injecting target critic helps the training or not. What I have described above is just my opinion why some popular implementations of AAC ignore target critic and still work. But honestly nothing stops us from adding it and see for ourselfs. My final opinion is that moving target problem is much less severe, but still exists, so injecting target critic can help.

Okay, now let's talk about the replay buffer. To understand if we can use it, we must remember the distinction between on-policy and off-policy algorithms. A replay buffer, by its nature, stores experiences collected by old versions of our policy, making it an off-policy technique.

So, are REINFORCE and Actor-Critic on-policy or off-policy? The policy gradient update rule is proportional to the gradient of the log-probability term - $\nabla_\theta \log \pi_\theta(a_t | s_t)$. This gradient is part of an expectation that is fundamentally calculated over the distribution of states and actions generated by the current policy, $\pi_\theta$.

Now, what happens if we use a replay buffer? The action $a_t$ in a stored transition was chosen by an old policy, not our current one. The core problem is a distribution mismatch: the data in our buffer comes from a different probability distribution than the one our current policy would generate. Using this off-distribution data to calculate the gradient leads to a biased estimate. It doesn't point in the correct direction to improve the performance of our current policy, which makes the naive off-policy update mathematically invalid.

Basically, imagine an old policy frequently took action $a_t = 1$ in state $s_t$, so our replay buffer now contains an overwhelming number of samples with this transition. Now, suppose our improved current policy has learned this is not a great action and assigns it a low probability, say $\pi(a_t = 1 | s = s_t) = 0.1$.

When we create our training batch, we sample 100 transitions from the replay buffer. Because of its history, let's say 60 of these samples are the transition $(s_t, a_t = 1)$.

Here's the mismatch: If we were acting on-policy, our current agent would only choose action $a_t = 1$ about 10 times out of 100 from that state. But our batch is telling a different story - it's showing up 60 times! The gradient update is an average over the entire batch. Because the action $a_t = 1$ is **over-represented by a factor of six**, the gradient component related to it - $\nabla_\theta \log \pi_\theta(a_t = 1 | s_t)$ - contributes six times more influence to the final update step than it should. This disproportionate weight completely biases the gradient, forcing our policy to update based on a distorted view of its own strategy. It's learning from the ghost of an old policy, not its present self.

For this reason, basic policy gradient algorithms are strictly on-policy, and a naive replay buffer is incompatible. But do not worry, there is a solution that can help us correct for this distribution mismatch. We will discover it in the next sections :)

## Advantage Actor-Critic

Okay, now we understand how we can utilize and improve a separate network to estimate the value of a given state. Now, the actor gradient update is proportional to $G_t - b(s_t)$, where the baseline is approximated via the critic $b(s_t) \approx V_\phi(s_t)$.

But that means that we are still have to wait until the end of episode to observe $G_t$. But wait a minute...

$G_t$ term indicates what return does action $a_t$ taken in state $s_t$ yield if we follow our policy afterwards. Does it sound familiar to you? It is exactly the definition of Q-value. So, it means that:

$$
G_t - b(s_t) = G_t - V^\pi_\phi(s_t) = Q^\pi(s_t, a_t) - V^\pi_\phi(s_t) = A^\pi(s_t, a_t)
$$

It is the definition of the advantage function, this is the same advantage we defined in Dueling DQN!

Okay, so with that being said, the actor's policy gradient becomes:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} A(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

Great, but still the problem is not solved, to estimate $A(s_t, a_t)$ we have to observe $G_t$, which means that we have to wait until the completion of an episode. We can't learn step by step like we did in DQN.

To be able to learn at each step we have to find a good estimate for a single step advantage. In other words, given a transition step $(s_t, a_t, r_{t+1}, s_{t+1})$ we need somehow to estimate $A(s_t, a_t)$ without the need to wait until the end of the episode to calculate the total returns. Let us look closely at the TD-target that we were computing for critic:

$$
\delta_t = r_{t+1} + \gamma V^\pi_\phi(s_{t+1}) - V^\pi_\phi(s_t)
$$

The TD-target - $r_{t+1} + \gamma V_\phi(s_{t+1})$ - can serve as a good single-step estimate of $Q^\pi(s_t, a_t)$, don't you think? Because it literally is. And that means that:

$$
\delta_t = Q^\pi(s_t, a_t) - V^\pi_\phi(s_t) = A^\pi(s_t, a_t)
$$

So, it turned out that the TD-target is actually a very good, one-step estimate of the advantage function, crazy right!

$$
A^\pi(s_t, a_t) \approx \delta_t
$$

---

Okay, I think we made a juge leap. Time to slow down a little. Let's recap once again what is the overall pipeline:

1. Actor, $\pi_\theta$, processes a state, $s_t$, and outputs a probability distribution over all actions
2. We sample from this distribution to choose an action, $a_t$ and we store the log-probability of that action, $\log \pi_\theta(a_t | s_t)$
3. We apply this action to the environment and we collect $(s_t, a_t, r_{t+1}, s_{t+1})$
4. Critic, $\phi$, processes $s_t$ and estimates the state-value: $V_\phi(s_t)$
5. We compute the TD-target as $y_t = r_{t+1} + \gamma V_\phi(s_{t+1})$
6. We compute the TD-error: $\delta_t = y_t - V_\phi(s_t)$

Now this single value, $\delta_t$, serves two purposes at once:

- For the critic: We square it to calculate the loss to apply a gradient descent on $\phi$:

$$
L_\phi  = \delta_t^2 \\
\phi \leftarrow \phi - \alpha_C \nabla_\phi L_\phi
$$

- For the actor: We compute actor's loss as TD-error times the log-probability and then apply a gradient descent on $\theta$ with a separate learning rate $\alpha_A$

$$
L_\theta = -\delta_t \cdot \log \pi_\theta(a_t \mid s_t) \\
\theta \leftarrow \theta - \alpha_A \cdot \nabla_\theta L_\theta
$$

And that is the Advantage Actor-Critic (AAC) algorithm. The critic learns the state-value function $V(s)$, and the actor uses the critic's TD-error as the advantage estimate to improve its policy at every single step!

We started with introducing a baseline with sole purpose of decreasing variance in $G_t$ and concluded in step-by-step learning with advantage estimate. And you know what else we did? We have solved the inherent problem of vanilla-PG of not being able to penalize bad actions when all returns are positive! The advantage, $A(s_t, a_t)$, shows the relative quality of the $a_t$, i.e. if that action is on average worse than others from this state. So, it means that if it worse, then the $A(s_t, a_t) < 0$, and by multiplying it with $\log \pi_\theta(a_t \mid s_t)$ we are moving in direction of decreasing the probability of taking $a_t$ from $s_t$. Incredible, right?

Okay, Advantage Actor-Critic is a beautifull algorithm. But we should not stop here. One of the obvious direction of improvement is to review whether we can extend one step learning to N-step. If you remember in previous lessons we had the same discussion that one step learning has much less variance but higher bias. I know that this whole section we have been trying to come up with a solution to decrease the variance, but high bias is also a problem. By introducing N-step learning we can find a sweat spot in this spectrum.

# N-step return

AAC works great, but as we already discussed one-step learning  has a high bias despite having low variance. In previous lessons we have seen that extending one step learning to N-step provides better results, can we adopt it to PG algorithms? Yes.

So, the formula we have seen is:
$$
\theta \leftarrow \theta + \alpha_A \cdot A(s_t, a_t) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

where $A(s_t, a_t)$ is estimated by the critic via TD-learning or more specifically TD(0) learning:

$$
A(s_t, a_t) = \delta_t \\
\delta_t =  r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Our goal is to extend the AAC to N-step learning. Greats news that we already know how to do it:

- First, let our actor interact with the environment for N steps, collecting a sequence of states, actions, and rewards. Let's say we start at time t:

$$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}, r_{t+2}, \ldots, s_{t+N-1}, a_{t+N-1}, r_{t+N}, s_{t+N})$$

- Then, we calculate the N-step return. Let's call it $G_{t:t+N}$ and it is the sum of discounted rewards for these N steps, plus the discounted value of the state $s_{t+N}$ estimated by our critic:

$$
G_{t:t+N} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{N-1} r_{t+N} + \gamma^N V_\phi(s_{t+N})
$$

**Important Note:** If your episode ends before N steps are completed (say it ends at step $k < N$, after reward $r_{t+k}$ and landing in a terminal state $s_{t+k}$), then the return calculation stops there, and $V_\phi(s_{\text{terminal}}) = 0$. For example, if $N=5$ but the episode ends at step 3 (relative to $t$):

$$
G_{t:t+3} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3}
$$


All right, choosing the optimal $N$ is another fine-tuning task that is going to be different from problem to problem. But if you remember that we had eligibility traces algorithm that was combining different N-step returns? We have something similar in PG algorithms called **Generalized Advantage Estimation (GAE)**. The core idea is to elegantly combine different N-step returns into one advantage estimate.

$$
A_t^{GAE} = \sum_{k=0}^\infin (\gamma\lambda)^k \delta_{t+k}
$$

**Important note**: $\delta_{t+k}$ represents here a single one-step TD-error at step $k$ or in other words:
$$
\delta_{t+k} = r_{t+k+1} + \gamma V_\phi(s_{t+k+1}) - V_\phi(s_{t+k})
$$

So, GAE is nothing else but a exponentially-weighted average of single-step returns at different steps.

- $\lambda = 0$: $A_t^{GAE} = \sum_{k=0}^\infin (\gamma \cdot 0)^k \delta_{t+k} = \delta_{t}$

    We assume that $0^0 = 1$, so if lambda is zero, then GAE is simply a single step return at current step
- $\lambda = 1$: $A_t^{GAE} = \sum_{k=0}^\infin \gamma^k \delta_{t+k}$ which is an infinite sum, but if we open up the equation that all $V_\phi(s_t)$ terms cancel out resulting in $ \sum_{k=0}^\infin \gamma^k r_{t+k+1} - V_\phi(s_t) = G_t - V_\phi(s_t)$.

    So if lambda is 1, then GAE is a monte-carlo return.

By choosing a value for λ between 0 and 1 (a common value is 0.95), we get a sophisticated blend of all possible k-step estimators. It gives more weight to the immediate TD-error but still incorporates information from many steps into the future, without being as noisy as a full Monte Carlo return.

## Practical Implementation

Usually, we do not wait for the whole episode to finish. We choose the hyperparameter $M$ which represents the number of steps the agent is willing to take before calculaing GAE and updating the networks.

- The agent collects a rollout (trajectory) by stepping M times in the environment. In addition to the usual tuple of experience we also collect $V_\phi(s_t)$ and $\log \pi_\theta(a_t \mid s_t)$

- Starting from the last step, $t = M$ and moving backward to the first, we use the following recursive logic
    - Calculate one step TD-error:
        - $\delta_t = r_t + \gamma \cdot V_\phi(s_{t+1}) - V_\phi(s_{t})$
        - $\delta_t = r_t - V_\phi(s_{t})$ if $s_{t+1}$ is terminal
        - Since we collected only up to $V_\phi(s_M)$ we have to explicitly calculate $V_\phi(s_{M+1})$
    - Calculate $A_t^{GAE}$
        - $A_t^{GAE} = \delta_t + \gamma \lambda A_{t+1}^{GAE}$
        - $A_t^{GAE} = \delta_t$ if $s_{t+1}$ is terminal
        - In the very beginning $A_{t+1}^{GAE} = 0$


This makes it very efficient to compute the GAE for every step in our collected trajectory. This GAE value is then used as the advantage estimate in the actor's loss function, leading to much more stable and effective policy updates.

One small detail about how to incorporate the target critic into GAE calculation. The pure theoretical implementation would be to not compute $V_\phi(s_t)$ using the online critic during collection of a rollout, but to store the state. Then inside the reverse loop, we would evaluate $s_t$ using online critic and $s_{t+1}$ using the target critic. However, it was found that only estimating $V(s_{M+1})$ using target network provide similar benefits. The reasons are:

- The final bootstrap value ($V(s_{M+1})$) is the "seed" of the entire recursive calculation. Stabilizing this one value provides the vast majority of the stability benefit.
- It requires minimal code changes and less computational overhead (fewer calls to the target network).

For these reasons, many high-quality implementations opt for the simpler approach.

# A3C

So far so good. We have started with a vanilla policy gradient theorem that stated that in order to improve the policy we should move in direction of
$$
G_t \cdot \nabla_\theta \log(\pi_\theta(a_t | s_t))
$$

Then we tried to battle the very high variance of each $G_t$ and concluded in advantage actor-critic that states that actually we should move in direction of

$$
A(s_t, a_t) \cdot \nabla_\theta \log(\pi_\theta(a_t | s_t))
$$

where $A(s_t, a_t)$ is approximated via $\delta_t$.

Then we said that we should not use only one-step TD error, $\delta_t$, for advantage approximation, but we should use Generalized Advantage Estimation that combines several N-step returns. So, actually to improve the policy we should move in the direction of

$$
A_t^{GAE} \cdot \nabla_\theta \log(\pi_\theta(a_t | s_t))
$$.

We have made a big step from from where we started, and in this section I would like to leave the PG update rule alone and I have a question for you.

Have you every watched Naruto? In a nutshell there is a character that can make copies of himself. Each copy has its own mind and can do things independently. After the copy expires, all learnt knowledge transfers to the original Naruto. And you know what? I think we can use Naruto's approach here as well!

Okay, so we train in the digital world. What stops us from making a bunch of independent agents that interact with the environment and gather data? This data can be then used to update the original, global policy. This way we:

1. Speed up training
2. Diversify the training data

Okay, so far so good. More data, more diverse data, faster training - sounds like a win-win-win. But we need to talk about why it might be making our training worse.

Remember how we talked about PG methods, like REINFORCE and AAC, being on-policy? That means they learn from data generated by the same policy that we are trying to improve. But it can be easily seen how this Naruto's approach can break this condition. Let's review how:

- We have a global policy, $\theta_{global}$, that all workers (clones) try to improve
- In the beginning each worker grabs a copy of the global policy parameters $\theta_{global}$ to its local brain $\theta_{local}$
- It then goes off into its own instance of the environment and collects a bunch of experiences $(s_t, a_t, r_{t+1}, s_{t+1})$ using its $\theta_{local}$.
- After a bit, it calculates gradients $\nabla \theta_{local}$ and sends them up to update $\theta_{global}$.

**But here’s the potential hiccup**: by the time a worker tries to update the $\theta_{global}$, it can be updated already by other workers, so it is not the same anymore. So, the global policy is already different from the $\theta_{global}$ the worker originally copied to its $\theta_{local}$! The policy it used to gather data, $\theta_{local}$, isn't exactly the same as the policy it tries to improve.

## Why A3C works

Despite that potential hiccup we discussed before, authors of Asynchronous Advantage Actor-Critic (A3C) used that Naruto's approach and showed impressive results. Why?

1. In A3C workers typically sync their $\theta_{local}$ with $\theta_{global}$ pretty frequently (e.g., after collecting a small batch of, say, 5 or 20 steps). So, the $\theta_{local}$ isn't ancient history; it's just a little bit behind the curve. The difference between the policy used for data collection and the global policy being updated is often small.

2. The massive benefit you get from all those workers collecting diverse, decorrelated experiences outweighs the off-policy scent. Remember our correlation problem with a single agent? We could not apply a replay buffer that helped us in DQN and I said that still there is a solution? The solution is A3C! A3C naturally solves this problem by having several agents gathering data. This rich, varied and uncorrelated data helps stabilize the learning process significantly.

3. Even if $\theta_{local}$ is a really old, the gradients computed from its experience are still likely to push $\theta_{global}$ in a beneficial direction. The environment dynamics don't usually change so drastically that what was good for a policy five steps ago is terrible for the current one.


A3C isn't strictly, 100% pure on-policy in the way that a single-threaded REINFORCE agent (that updates only after a full episode with no other policy changes) would be. There's a touch of off-policy flavor because of the asynchronous updates from multiple workers. But it works because the policies don't diverge too dramatically between syncs, and the immense benefits of parallel, decorrelated data collection provide a much more stable and efficient learning signal overall. The global policy is learning from the collective, slightly time-delayed wisdom of its many clones.

If you are confused about why do we really care that the algorithm is on-policy or off-policy, let's discuss it.

---

## On-Policy vs Off-Policy

Okay, let's introduce some terms for better and clearer understanding. Our policy $\pi_\theta$ is the brain that decides what actions it should take given that it is in some state. To train an RL agent, we need to gather data. To gather data, the agent has to interact with the environment, which means picking actions, performing them, and observing the reward and the resulting state. When the agent is gathering data, it is using a "behavior policy" - in other words, the version of its $\pi_\theta$ that was used to collect the data.

After data is collected, say a single $(s_t, a_t, r_{t+1}, s_{t+1})$ tuple, we want to use it to improve our policy. The policy that we are going to improve we will call the "target policy" - the version of our $\pi_\theta$ that we are going to update (via backpropagation, for example).

An algorithm is considered on-policy if the target and behavior policies are the same. In other words, the agent updates the same version of $\pi_\theta$ that it used to gather the data. We have seen this in REINFORCE, where we used the same policy for data gathering and updating.

An algorithm is considered off-policy if the target and behavior policies are not guaranteed to be the same. In other words, the agent updates its current $\pi_\theta$ with data gathered by an older version of itself. We saw this in our DQN chapter, where we gathered data, stored it in a replay buffer, and then sampled from that buffer to update the network.

**Now, why should we care so much about this distinction?**

- The first reason is that off-policy tools, like replay buffer, does not work with on-policy algorithms

    We have discussed already why replay buffer is incompatible with AAC or REINFORCE.

- The second reason is that on-policy algorithms have the unique property of higher stability.

    The stability comes from the fact that the update is always directly relevant to the current policy. The gradient is a low-variance, "true" signal for improvement. This property makes on-policy algorithms much more stable than their counterparts. Off-policy algorithms can be unstable because learning from "stale" data can be dangerous. The algorithm has to correct for the fact that the old data wasn't generated by the current policy. This is why state-of-the-art algorithms like SAC and TD3 need so many tricks (twin critics, target networks) to avoid divergence.

This sounds like on-policy methods are the obvious choice, but then why would anyone use off-policy methods? The answer is the single most important trade-off in modern RL: sample efficiency.

**On-Policy is Inefficient:** Because an on-policy algorithm must throw its data away after every single update, it is constantly demanding fresh interactions with the environment. This can be incredibly slow and expensive. Imagine a real-world robot having to perform a task thousands of times just to make one small policy update.

**Off-Policy is Efficient:** An off-policy algorithm's replay buffer is its superpower. It can learn from a single experience hundreds or even thousands of times. By sampling from this large and diverse bank of past experiences, it can squeeze much more learning out of every interaction. This makes it far more suitable for robotics and other real-world tasks where collecting data is costly.

---

Okay, with that being said, we see that A3C is not quite on-policy, but what if I told you that it can be made on-policy and at the same time much simpler?

## A2C and A3C

Okay, so how A3C can be made strictly on-policy and simpler at the same time? Easy, we just replace asynchronous update with synchronous.

Parallel learning was initially introduced with an asynchronous model, Asynchronous Advantage Actor-Critic (A3C). The idea was to have N "worker" agents, each with its own copy of the model and interacting with its own copy of the environment. Then, each worker would independently compute gradients and send them back to update one global, master model.

That approach, while clever, was found to be suboptimal for a few key reasons, which led to the development of a simpler, synchronous version: Advantage Actor-Critic (A2C) (Okay I know that we used AAC term for that, but it was just because I did not want to mix synchronous A3C with Advantage Actor-Critic formulation, from now on we will use A2C term).

Let's break down why A2C is generally considered an improvement.

1. **Make the model strictly on-policy**

As we discussed, in A3C, workers operate completely independently. Imagine Worker #1 starts its work with version 10 of the global model's weights. It spends some time collecting its M experiences. In the meantime, Workers #2, #3, and #4 might have already finished their work and sent their own updates. By the time Worker #1 sends its gradients, the global model might be on version 13.

Worker #1's gradient, which was calculated based on version 10, is now "stale." Applying this outdated gradient to the newer version 13 of the model is inefficient and can lead to unstable learning. The master model is essentially being pulled in slightly different directions by workers who are looking at slightly different versions of the "map."

2. **GPU Efficiency and Batching**

This is the most significant practical reason for A2C's dominance.

A3C is built for CPUs. Its strength is using many CPU cores, with each worker sending a small, frequent update. GPUs, however, achieve their incredible speed by performing the same operation on thousands of data points at once in massive parallel batches. The small, individual gradients sent by A3C workers are highly inefficient for a GPU to process.

A2C is built for GPUs. In A2C, we wait for all N workers to finish their M steps. We then gather all N * M experiences into one large batch. The central model can then perform a single, massive forward and backward pass on this large batch, taking full advantage of the GPU's parallel processing power. The performance gain from this efficient batching almost always outweighs the time spent waiting for the workers to synchronize.

3. **Simplicity and Stability**

The A2C approach is simply easier to implement and more stable. Instead of managing complex asynchronous processes, locks, and shared optimizers, you can use a Vectorized Environment (VecEnv). This object handles the parallel environments for you. The main agent code interacts with it as if it were a single environment that just happens to accept a batch of actions and return a batch of observations.

This synchronous "collect-then-update" cycle leads to a more stable gradient estimate because it's based on a consistent, non-stale snapshot of the policy across all workers.

In summary, while A3C was a revolutionary idea that showed the power of parallel data collection, A2C quickly became the standard because it better aligns with modern hardware (GPUs) and offers a more stable and simpler implementation.

# A2C

Okay, so how does A2C actually work in practice? The key is to stop thinking about individual workers and instead think about operating on batches of data at every step.

First, we need a new object that wraps our N copies of the environment. This object, often called a Vectorized Environment (`VecEnv`), is the magic behind A2C's simplicity. From the agent's perspective, it looks like a single environment, but every time you call `step()`, it sends one action to each of its N internal environments and returns a batch of N observations.

With this tool, we can define a clear, synchronous pipeline. Let's use an analogy: think of A2C as a perfectly synchronized rowing team.

- The Rowers: The N parallel environments.

- The Coxswain: The single Actor-Critic agent.

The entire process can be broken down into two distinct phases: data collection and learning.

## Phase 1: The Rollout

In this phase, the agent's only job is to act according to its current policy and gather a rich batch of experiences. No learning happens yet.

1. **Initialize**

    The coxswain gets the initial state from all N rowers at once. This is a batch of states with shape `[N, observation_dimension]`.

2. **Collect M-Step Trajectories**

    We now run a loop for a fixed number of steps, $M$ (e.g., $M=20$), to collect a "rollout".

3. **Get Actions**

    The coxswain (agent) looks at the current batch of $N$ states and, using the Actor network, decides on a batch of $N$ actions to take.

4. **Step the Environments**

    The coxswain shouts the command, and all $N$ rowers (environments) take their assigned action in unison by calling `env.step()`.

5. **Gather Feedback**

    The agent receives a batch of feedback: $N$ next states, $N$ rewards, and $N$ done flags.

6. **Store Everything**

    The agent stores all the information from this step - the states, the actions taken, the rewards received, and critically, the log-probabilities of the actions $\log \pi(a|s)$ - in a temporary buffer.

After looping $M$ times, our buffer now contains $M$ sets of $N$ transitions, which can be thought of as a large collection of $N \times M$ experiences.

## Phase 2: The Update

Now that the rollout is complete, the entire team stops rowing. It's time for the coxswain (agent) to review what just happened and update the team's strategy. This happens in a single, large, synchronous update.

1. **Calculate Returns and Advantages**

    The agent looks at the $N \times M$ experiences it just collected. It calculates the discounted return $G_t$ and the advantage estimate $A_t \approx \delta_t$ for every single step in the buffer. By looking ahead $M$ steps, it gets a much more stable and less biased estimate of the advantage compared to just looking one step ahead.

2. **Compute Losses**

    Using this batch of advantages and the stored log-probabilities, the agent computes a single, aggregate loss value for the Actor and a single loss value for the Critic.

3. **Apply Gradients**

    Finally, the agent calls `.backward()` on these losses and takes one single optimization `step()` to update the weights of its Actor and Critic networks.

Once the update is complete, the new, improved strategy is ready, and the entire process repeats, starting with a new rollout from Phase 1.

Now because we have just one policy and all the parallel environments step in the synchronous manner, the data gather is on-policy and we do not need to care about PG theorem not working! And more importantly we just made the learning pipeline much simpler to implement.

# Policy Update Stability

All right, we have seen that using parallel environments we can improve the efficiency of training. But still, the main idea behind learning remains: if an action led to a good outcome, make it more likely; if it led to a bad outcome, make it less likely.

$$
\theta \leftarrow \theta + \alpha \cdot (\text{Some\_Goodness\_Score}) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

This is great, but may be you have noticed this behavior: You train a policy for some time and it seems to converge, the total reward is stably high; but suddenly the policy started to deteriorate, the overall reward is lower than it was. This is happening because each update can lead our policy to some weird directions and one unfortunate big step can move the policy from the its stable version.

A large, possibly noisy update could accidentally overwrite good parts of the learned policy. The agent might "forget" previously learned skills because of one aggressive update based on a potentially unrepresentative batch of data. In LLM fine-tuning people have same problem, and it is called "Catastrophic Forgetting"

This issue is often referred to as the problem of destructive policy updates or instability due to large policy ratios. The "policy ratio":
$$
\frac{\pi_{\theta_\text{new}}(a|s)}{\pi_{\theta_\text{old}}(a|s)}
$$
tells us how much the probability of taking an action changes. If this ratio becomes too large or too small for actions that had a significant impact, the update can be unreliable.

And then the question was raised, can we limit the amount of update? We still want our policy to change, but may be we can define a trust region around the current policy and we don't apply changes that lead to our new policy outside of this region?

## Early Attempts to Control Update Size: Trust Regions

Recognizing this problem, researchers started thinking about ways to ensure that the new policy doesn't stray too far from the old one. The core idea was to define a "trust region" around the current policy $\pi_{\theta_\text{old}}$. We only want to update our policy to a $\pi_{\theta_\text{new}}$ that stays within this trusted region, where we believe our data and advantage estimates are still reasonably valid.

One of the most well-known algorithms that formalized this was **Trust Region Policy Optimization (TRPO)**. TRPO aimed to maximize the policy gradient objective while explicitly constraining the "size" of the policy update, usually measured by the Kullback-Leibler (KL) divergence between $\pi_{\theta_\text{old}}$ and $\pi_{\theta_\text{new}}$. The KL divergence $D_\mathrm{KL}(\pi_{\theta_\text{old}} \| \pi_{\theta_\text{new}})$ is a measure of how different two probability distributions are.

TRPO would try to solve something like:

$$
\text{Maximize } \mathbb{E}[\ldots] \text{ subject to } D_\mathrm{KL}(\pi_{\theta_\text{old}} \| \pi_{\theta_\text{new}}) \leq \delta_\mathrm{KL}
$$

(where $\delta_\mathrm{KL}$ is a small constant defining the size of the trust region).

While TRPO was very effective and produced stable learning, it had a downside: it was relatively complex to implement. Solving that constrained optimization problem often involved approximations and second-order optimization methods, which are not as straightforward as typical gradient descent.


## Simplicity and Stability: PPO

This complexity of TRPO motivated the search for algorithms that could offer similar stability benefits but with a simpler implementation, ideally using only first-order optimization (like the gradient ascent/descent we're used to).

And this is precisely where **Proximal Policy Optimization (PPO)** comes into the picture. PPO algorithms are designed to keep the new policy close to the old policy, thereby preventing destructively large updates, but they do so using clever modifications to the objective function that are much easier to implement and compute.


## Proximal Policy Optimization

Alright, remember how we're trying to update our policy $\pi_\theta$? When we do an update, we're essentially trying to find a new set of parameters $\theta_{\text{new}}$ based on data we collected using the old policy parameters, let's call them $\theta_{\text{old}}$. So, $\theta_{\text{old}}$ are the parameters of the policy that actually went out, played the game, and brought back a batch of experiences $(s_t, a_t, r_{t+1}, s_{t+1}, \ldots)$.

And in its foundation PPO just wants to know, for a given state $s_t$ and an action $a_t$ that we actually took (using $\pi_{\theta_{\text{old}}}$): "How much more or less likely is it that our new policy $\pi_\theta$ would have taken that same action $a_t$ in that same state $s_t$?"
Because if the same action gets much higher probability than before, it means that the policy is changing too much. And we don't want that.


This question can be measured numerically with $r_t(\theta)$. It's defined as:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

Let's break this down:

- $\pi_\theta(a_t|s_t)$ (the numerator): This is the probability of taking the specific action $a_t$ in state $s_t$ according to our new policy (imagine like we already calculated gradients from these samples and updated our policy).
- $\pi_{\theta_{\text{old}}}(a_t|s_t)$ (the denominator): This is the probability of taking that same action $a_t$ in that same state $s_t$ according to the old policy that was actually used to generate the experience.

What does this ratio tell us?

- If $r_t(\theta) > 1$: This means our new policy $\pi_\theta$ makes action $a_t$ more likely in state $s_t$ than the old policy did. For example, if $\pi_{\theta_{\text{old}}}(a_t|s_t) = 0.2$ and $\pi_\theta(a_t|s_t) = 0.4$, then $r_t(\theta) = 2$. The new policy is twice as likely to pick that action.
- If $r_t(\theta) < 1$: This means our new policy $\pi_\theta$ makes action $a_t$ less likely in state $s_t$. For example, if $\pi_{\theta_{\text{old}}}(a_t|s_t) = 0.5$ and $\pi_\theta(a_t|s_t) = 0.25$, then $r_t(\theta) = 0.5$. The new policy is half as likely to pick that action.
- If $r_t(\theta) = 1$: The probability of taking action $a_t$ in state $s_t$ hasn't changed between the old and new policies.

The ratio $r_t(\theta)$ is basically a direct measure of how much our policy is changing for the specific actions that we've seen and are trying to learn from.
If this ratio gets too far from 1 (either too large or too close to 0), it signals that our new policy is becoming significantly different from the policy that collected the data. So, PPO is using this ratio as a safety monitor to prevent radical changes to our policy.

If you are confused at this point, let's digest an important point of PPO. You might think, **how is it possible that we have old and new policies in on-policy algorithms?**

This is where things get interesting. In a strict on-policy algorithm like A2C, you're right, we don't really have a notion of an "old" and "new" policy; we collect data, update once and then discard this data. But another PPO's key innovation in addition to safe updates is data efficiency. The authors of PPO said, what if we were to re-use the sample that we just gathered and not discard it once we found gradients? It is this reuse that creates the need for the old vs. new policy distinction within an on-policy framework.

This is different from truly off-policy methods like DQN, which use a large Replay Buffer to store data gathered by many past versions of the policy. Proximal Policy Optimization uses the importance sampling ratio, yet it's still considered an on-policy algorithm. So, before discussing how PPO limits the updates to the policy, let's see how that's possible!

### PPO's sample efficiency

First, let's rewind for a sec. In traditional policy gradients, the core idea is that if an action $a_t$ taken in state $s_t$ leads to a positive advantage $\hat{A}_t$, we want to increase the probability of taking $a_t$ in $s_t$. If $\hat{A}_t$ is negative, we want to decrease that probability. The update often looks something like:

$$
\text{Change in policy} \propto \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

However, PPO takes a slightly different, but related, route to formulate its objective function using our new friend, the probability ratio $r_t(\theta)$. It starts with a "surrogate" objective function. This basic surrogate objective is:

$$
J_{\text{SURROGATE}}(\theta) = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t]
$$

Here, $\hat{\mathbb{E}}_t[\ldots]$ means we're taking the average over all the time steps $t$ in our batch of experiences. We want to choose $\theta$ to **maximize** this $J_{\text{SURROGATE}}(\theta)$.

### Why this surrogate function works?

This is a very sharp question. How can we just swap our objective from the total return $J(\theta)$ to this new formula $r_t(\theta) \cdot \hat{A}_t$? The connection is not obvious, but it is mathematically sound, and it's built on a statistical technique called **Importance Sampling**.

The standard PG has a major limitation: it's strictly **on-policy**. This means the data (the actions and advantages) used to calculate the gradient must have been collected using the exact same policy $\pi_\theta$ that we are trying to improve.

$$
\nabla J(\theta) = \mathbb{E}[\hat{A}_t \cdot \nabla \log \pi_\theta(a_t|s_t)]
$$

On the contrast, PPO is designed to be more sample-efficient. We want to collect a batch of data with an old, fixed policy $\pi_{\theta_{\text{old}}}$ and then update our current policy $\pi_\theta$ for several steps using that same data. This is an **off-policy** setup.

So the central question is: **How can we evaluate our new policy $\pi_\theta$ using data generated by the old policy $\pi_{\theta_{\text{old}}}$?** Remember Prioritized Experience Replay and what we did to solve similar problem?

### Importance Sampling

**Importance Sampling** provides the mathematical bridge. It's a technique that allows us to calculate the expected value of a function under a new probability distribution $p$ using samples drawn from an old distribution $q$. The formula is:

$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
$$

Let's map this directly to our goal:

- We want to find the expectation of our advantage function, $\hat{A}_t$, using samples generated by our current policy, i.e. $p = \pi_\theta$
- But using IS we can find the same expectation but with samples generated not by $p$ or $\pi_\theta$, but by $q$ or $\pi^{old}_\theta$

Plugging these in, the objective function $J(\theta)$ for the new policy is:

$$
J(\theta) = \mathbb{E}_{a_t \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t\right]
$$

And this is how the PPO surrogate objective was derived!

$$
J_{\text{SURROGATE}}(\theta) = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t]
$$

This shows that our surrogate objective is a valid way to estimate the performance of the new policy, at least from IS perspective. But you already know that for this function to be a valid surrogate, then we have to check the gradients.

### Gradient of surrogate loss function

The final check is to ensure that the gradient of our new surrogate objective is related to the original policy gradient we started with. Let's find the gradient of $J(\theta)$. Remember that $\hat{A}_t$ and $\pi_{\theta_{\text{old}}}$ are just constants from our collected data; the only variable is $\theta$ in $\pi_\theta$.

$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_t [r_t(\theta) \hat{A}_t] = \mathbb{E}_t [\hat{A}_t \cdot \nabla_\theta r_t(\theta)]
$$

Using the log-derivative trick ($\nabla f = f \cdot \nabla \log f$), we get:

$$
\nabla_\theta r_t(\theta) = r_t(\theta) \cdot \nabla_\theta \log r_t(\theta) = r_t(\theta) \cdot \nabla_\theta \log \left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\right) = r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$
The term involving $\pi_{\theta_{\text{old}}}$ disappears because its gradient with respect to the current policy parameters $\theta$ is zero.

Substituting this back into our gradient equation gives:

$$
\nabla_\theta J(\theta) = \mathbb{E}_t [\hat{A}_t \cdot r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)]  \tag{5}
$$

Okay, this formula look quite similar to the equation (4), the only difference is this term $r_t(\theta)$. To understand why the gradient is unaffected by this term, we have to understand how exactly learning happens in PPO.

Here is the simple pipeline:

1. We gather data (rollouts) using our policy $\pi_\theta$.

    Let's say we are using 1-step advantage: $(s_0, a_0, r_1, s_1, \pi_\theta(a_0|s_0), \log \pi_\theta(a_0|s_0))$

2. We use that data to calculate gradients and update the policy to obtain a $\pi_\theta^{new}$.

    The whole trick is to let us train not only one time on this sample, but as much as we want. Let say we want to train our network two times on this sample. We calculate the gradient using formula (5) for the first time. What is the value of $r_t(\theta)$? It is exactly 1! Why? Because we have not changed the policy yet, it is only the first step of the update. So, in the beginning our ratio is equal to 1.

    We start to calculate losses and update the networks:
    - TD-error: $\delta_0 = r_1 + \gamma \cdot V_\phi(s_1) - V_\phi(s_0)$
    - Actor loss: $L_a = -\delta_0 \cdot 1 \cdot \nabla_\theta \log \pi_\theta(a_0|s_0)$
    - Critic loss: $L_c = \delta_0^2$
    - Run backpropogation and get updated weight and $\pi_\theta^{new}$

    In the first optimization step, the gradient of our surrogate loss is exactly the same as the original. That makes this loss a valid proxy at least in the first step.

3. Now we want to re-use that data to update the $\pi_\theta^{new}$ again.

    So, now after the first training step, our policy has changed. It is no longer the same as it was when this sample was collected, but we still want to re-use it for training. In strict on-policy algorithms it is impossible, but due to importance sampling we can do it here!

    Our current $\pi_\theta^{new}$ does not equal anymore to its version that we used for data collection, $\pi^{old}_\theta$. Naturally, value of $\nabla_\theta \log \pi_\theta(a_0|s_0)$ is not correct anymore with respect to the new policy. So, it has to be computed again.
    - Find the probability of picking action $a_0$ from $s_0$ using $\pi_\theta^{new}$ and calculate its logarithm
    - Similarly, our critic changed as well, so we have to recalculate TD-error: $\delta_0 = r_1 + \gamma \cdot V^{new}_\phi(s_1) - V^{new}_\phi(s_0)$

    Now the actor loss equals to $-\delta_0 \cdot r(\theta) \cdot \nabla_\theta \log \pi_\theta^{new}(a_0|s_0)$, it is clear that every term in this loss has changed from the previous step. Does this gradient still point in the correct direction?

    Yes. The new critic knows better (because it was updated) whether the action was good or bad (whether advantage is positive or negative), so the gradient still points in the correct direction.

    **Then why do we need this ratio?** This ratio is basically needed to correctly scale this update.

---

A good way to build an intuition about why this ratio really helps is to look at the concrete example:

Imagine there is a good action $a_1$ in some state and imagine our policy does not yet know that it is good, so $\pi(a_1) = 0.05$. But because we use sampling, it is clear that this action can be sampled, and imagine we were lucky and this action indeed was sampled.

Now, the critic estimated a huge advantage, let's say $A = 10$ for that action. In the first iteration we updated our policy proportional to $10 \cdot \nabla_\theta \log(0.05)$ and let's say the new policy is smarter now and assigns $\pi(a_1) = 0.1$.

So, we want to re-use that sample to update already updated policy, without the ratio that update would be proportional to $10 \cdot \nabla_\theta \log(0.1)$ (another assumption that critic still estimates the advantage as 10)

But now let's see what happens when we apply the importance sampling ratio. The ratio is:

$$
r = \frac{0.1}{0.05} = 2
$$

The corrected update is now proportional to $r(\theta) \cdot A \cdot \nabla_\theta \log{\pi^{new}(a_1)}$, which becomes $2 \cdot 10 \cdot \log(0.1)$. The update is now twice as large.

This is the key insight. The ratio is correcting for the fact that the data was generated by a worse policy. It reasons that the event-sampling the rare but excellent action $a_1$ - was a lucky fluke under the old policy. We gathered only one sample, but what if we used a mini-batch of size 256? How many samples of $a_1$ would be present in this mini-batch? Well, it would be something around 13. So, it means without the ratio the policy would have been updated 13 times by the factor of $10 \cdot \nabla_\theta \log(0.1)$.

But this is incorrect for our new, smarter policy. The policy after the first update understands that this action is twice as likely and therefore twice as important. It means that if this policy went and gathered a mini-batch of experience, then there would be something around 26 samples of $a_1$. The old data has under-sampled this crucial event, so to get an accurate gradient for our new policy, we must up-weight its importance. The ratio ensures that the update magnitude correctly reflects how representative the action is of the current policy, not the outdated one that collected the data.

---

So, gradients are mostly the same, the only difference is the ratio. But the ratio is essential to correctly scale the gradients to correct for the fact that our policy changes. It gives a massive upgrade to the last algorithm we reviewed, A2C. We are no longer just discarding data after only one update, we can run as much as we want!

I hope it got a bit clearer what is that new and old policies and why we care so much about them. Sounds good, right? But initially we were discussing the "dangers of taking big steps"! So, **what happens if the policy update tries to make $r_t(\theta)$ huge?**

### PPO's safe policy update rule

So, now that we understand better how we can re-use old samples for training, a.k.a. mix off and on policy, using the ratio, let's talk about how the same ratio can help us prevent dangerous, radical updates in the policy.

Imagine $\hat{A}_t$ is positive and large. The optimization process might try to make $\pi_\theta(a_t|s_t)$ extremely large (close to 1) and assume that $\pi_{\theta_{\text{old}}}(a_t|s_t)$ was small, leading to a massive $r_t(\theta)$. This could result in an enormous update step, potentially overshooting and destabilizing our policy. So, the ratio that we compute for IS is actually is a good measure of safety of the update. But the problem is that this unconstrained surrogate objective $J_{\text{SURROGATE}}(\theta)$ doesn't have any built-in mechanism to prevent these overly large policy changes. It doesn't explicitly stop $r_t(\theta)$ from going too far from 1. If we just try to maximize this, we could run into the same instability problems we wanted to solve!

Luckily for us, the autors of PPO came up with a very clever and simple solution. Their solution was just to clip this objective! More specifically, the main PPO's idea is: "What if we let the ratio $r_t(\theta)$ do its thing, but only within a certain safe range? If it tries to go outside this range, we'll just... well, clip it!"

1. **Defining the "safe zone" with epsilon**

PPO introduces a small hyperparameter, usually called $\epsilon$ (epsilon), typically set to something like 0.1 or 0.2. This $\epsilon$ defines our "trust region" or "allowed deviation" from a ratio of 1.
The idea is that we're generally okay with the new policy being, say, 10% to 20% more or less likely to take an action than the old policy ($1 \pm \epsilon$). Beyond that, we get suspicious.

2. **The Clipped Ratio**

We take our probability ratio $r_t(\theta)$ and clip it:

$$
\text{clipped\_ratio}_t(\theta) = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

This clip function does exactly what it sounds like:

- If $r_t(\theta)$ is between $1-\epsilon$ and $1+\epsilon$, then $\text{clipped\_ratio}_t(\theta)$ is just $r_t(\theta)$.
- If $r_t(\theta)$ is greater than $1+\epsilon$, then $\text{clipped\_ratio}_t(\theta)$ becomes $1+\epsilon$. (It can't go higher).
- If $r_t(\theta)$ is less than $1-\epsilon$, then $\text{clipped\_ratio}_t(\theta)$ becomes $1-\epsilon$. (It can't go lower).

3. **The PPO Clipped Surrogate Objective $L_{\text{CLIP}}(\theta)$**

Now, PPO crafts its objective by taking the minimum of two terms:

- The original surrogate objective: $r_t(\theta) \hat{A}_t$
- The surrogate objective using the clipped ratio: $\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\cdot \hat{A}_t$

So, the full PPO clipped surrogate objective function is:

$$
J_{\text{CLIP}}(\theta)= \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon) \hat{A}_t \right) \right]
$$


### How $J_{\text{CLIP}}(\theta)$ actually works?

By taking the minimum of the "normal" objective and the "clipped" objective, PPO essentially creates a lower bound (or a pessimistic bound) on how much the policy can be changed in one go.

This clipping mechanism is surprisingly simple yet incredibly effective at stabilizing training. It ensures that the policy changes in a more gradual and controlled manner, making PPO much less sensitive to hyperparameter choices and more robust overall compared to older policy gradient methods, like A2C or REINFORCE.

---

One hidden detail for people without big expertise in deep learning is how exactly does PPO prevents the radical policy updates. If a data sample from batch results in ratio > $1+\epsilon$ and positive advantage, then the clip becomes active and replaces the $r_t$ term with a constant $1+\epsilon$. Because the objective function is now a multiplication of two constants, then it means that the gradients are all zero. Or in other words, the data samples that forces the algorithm to clip the ratio contribute no effect to weight updates.

---

Now to build a deeper understanding of the philosophy behind PPO, let's research different case-studies of this clipped objective. For this example, let's set epsilon to 0.2.

- Advantage is positive, for example $A = 1$, i.e. good action was taken
    - Ratio is normal, for example $1.1$. So, clip operator is not triggered, meaning a normal gradient flow.
    - Ratio is huge, for example $2.0$. So, naturally $r$ is clipped to $1.2$. And because $2 \cdot 1 > 1.2 \cdot 1$ the min operator chooses the second term. No gradients are tracked and no update is applied.
    - Ratio is small, for example $0.5$. So, naturally $r$ is clipped to $0.8$. And because $0.5 \cdot 1 < 0.8 \cdot 1$ the min operator chooses the first term. Gradients are tracked and update is applied
- Advantage is negative, for example $A = -1$, i.e. bad action was taken
    - Ratio is normal, for example $0.9$. So, clip operator is not triggered, meaning a normal gradient flow.
    - Ratio is huge, for example $2.0$. So, naturally $r$ is clipped to $1.2$. And because $2 \cdot -1 < 1.2 \cdot -1$ the min operator chooses the first term. Gradients are tracked and update is applied.
    - Ratio is small, for example $0.5$. So, naturally $r$ is clipped to $0.8$. And because $0.5 \cdot -1 > 0.8 \cdot -1$ the min operator chooses the second term. No gradients are tracked and no update is applied.


Okay, so this was confusing to me at first, why in some cases PPO allows for an update and in other cases don't. Let's build an intuition why:

1. **$r$ is in normal range with positive advantage**

Okay, so this is exactly what we want. After some optimization steps, we see that the current version of our actor (policy) thinks that this action is good (its probability is higher), but the probability does not go that far from what it was during policy that gathered that experience sample. So, reinforcing this good action even more is still within our safe zone, this will only imporve the policy and won't accidentally change it too much.

2. **$r$ is in normal range with negative advantage**

Okay, this is similar to the first case, but the action was actually bad. The current policy assigns a lower probability to this action in comparison to the old policy that gathered the data. However, the difference between these probabilities is not big, so we should be good by running another update step to penalize this bad action more.

3. **Ratio is huge with positive advantage**

Now this case is what we want to avoid. The agent discovered a good action, but we already updated our policy so much that now it assigns much higher probability to this action.

However, what if that was just a random fluke? Or our advantage estimate was super optimistic and not true? Or assigning even higher probability to this action will limit the amount of exploration that we should still do? In short there is no 100% guarantee that this action is the best among all others. The current version of the policy already assings much higher probability, increasing it even higher can in the result be risky and de-stabilize our training. The best play would be to play safe and don't apply the gradient descent.

4. **Ratio is small with positive advantage**

Okay, this might be a weird case. So, the agent discovered a good action and for some reason the current policy assings a lower probability to it than the old policy did?

First of all let's understand how is that even possible.

Well, a good action means that the critic estimates its advantage as positive. Positive advantage times the log probability will always push the policy towards increasing the probability of that action, so theoretically, if we were to train only on one sample, then this scenario would not be possible.

However, this might happen when we train our network on some mini-batch (let's say of size 128). In this case, the gradient that updates the policy's weights is the average of the gradients from all samples in the mini-batch (e.g., 128 different experiences). Each sample in the batch "pulls" the network's weights in a different direction.

- A sample $(s_1, a_1, A > 0)$ wants to change the weights to increase the probability of $\pi(a_1 | s_1)$

- A different sample $(s_2, a_2, A > 0)$ wants to change the weights to decrease the probability of $\pi(a_2 | s_2)$.

These desired changes might require modifying the same weights in opposite ways. The final gradient vector applied to the weights is a compromise that improves the policy's performance on average across the entire batch. This compromise update, while beneficial for the batch as a whole, can have localized, counter-intuitive effects. For one specific sample $(s_k, a_k)$ that had a positive advantage, it is mathematically possible that the overall weight change - which was dominated by the "pulls" from the other 127 samples - coincidentally caused the probability $\pi(a_k | s_k)$ to decrease.

Now that we know that it is very much possible, we have to understand that is exactly the opposite of what we want, we want to increase probability of good actions. So this mistake must be corrected.

From one hand, PPO could just easily discard this sample and also do not do any updates. But if we do nothing then how this mistake is going to be corrected? The current policy is already, for some mistaken reasons, assigns a lower probability to this good action that it was before. If we reject this sample, who is going to say to our policy: no, you should do the opposite, you should increase its probability? We must explicitly try to move the policy in the opposite direction to correct its behavior.

- So, PPO calculates the gradient of the objective: $\nabla_\theta L(\theta) = \mathbb{E}_t [\hat{A}_t \cdot r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)]$
- As we already said before the log probability term is the vector pointing in increase of likelihood of that action.
- This vector is scaled by this positive number $\hat{A}_t \cdot r_t(\theta) = 0.5$
- The final parameter update becomes: $\theta_{new} \leftarrow \theta_{old} + \alpha \cdot 0.5 \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$
- That means that even if for some reason our new policy tries to make the good action less probable, we will still make a step towards increasing its likelihood to try to fix it.

5. **Ratio is small with negative advantage**

It means that the agent discovered a bad action and the current policy makes it very less probable.

The same logic applies as to the first case, what if that was a mistake? The policy already learned to make it much less probable, adding more momentum in this direction can be risky.

Now this might be contradicting your intuition. The scale of $\nabla_\theta \log \pi_\theta(a_t|s_t)$ is not huge, in our example it is $-0.5$. This means that this is a small step, but still PPO rejects this update, why?

---

**Very important sidenote**

Now here I had a confusion when I first read about PPO. I happened to build an impression that PPO tries to limit the magnitude of the update, i.e. the gradient or the advantage times ratio. And I think that I might have given you the same impression. It turned out to be not true. Honestly, if we think about it, it could be easily done by decreasing the learning rate, because it directly scales the magnitude, or via gradient clipping, a tool that practioners often use.

PPO was designed by smart people, and they were looking much deeper. What PPO is actually trying to constrain is the change of policy. Even a small update to neural network's parameters can cause drastic changes in the policy, so PPO shouldn't control the magnitude, but the probability distribution over actions.

Imagine you had your policy of always discovering first the left room for treasures and then proceeding to the right room for finishing the game. Now, after an unlucky parameter update your policy has changed, such that you now don't enter the left room at all. You experience change dramatically and this is what PPO tries to avoid. It does not care whether the magnitude, the advantage times ratio, is big or small; all it cares about that probability distribution over actions of the new policy is not very different from probability distribution over actions of the old policy. And it does that via clip operator.

So, it can allow very big learning steps with only condition that the probability of a certain actions does not go outside the safe zone.

---

6. **Ratio is huge with negative advantage**

This case represents an experience when a bad action is made more probable. This is similar to case 4. This is the mistake that we must correct.

In support of the claim that PPO is not trying to constraint the magnitude of the update, in this specific case an update would be:

$$
\theta_{new} \leftarrow \theta_{old} + \alpha \cdot -2 \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

The learning step is of magnitude of -2, and PPO does not mind. And all because the new probability does not diverge too much from the old one.

---

This is a summary table of all possible cases:

| Advantage | Ratio r | Clipped? | Gradient? | Intuition|
|-----------|---------|----------|-----------| ---------|
| A > 0 | 0.8 < r < 1.2 | No | Yes | Normal improvement |
|A > 0 | r > 1.2 | Yes | No | Prevent overly optimistic updates|
|A > 0 | r < 0.8 | Yes | Yes | Correct mistaken probability decrease|
|A < 0 | 0.8 < r < 1.2 | No | Yes | Normal improvement|
|A < 0 | r > 1.2 | Yes | Yes | Correct harmful mistake|
|A < 0 | r < 0.8 | Yes | No | Already improved enough|

### Final Loss

Typically, the $J_{\text{CLIP}}(\theta)$ is not the only part of the loss. The full loss function that PPO implementations often optimize (minimizes) includes:

- The actor loss - $L_{\text{CLIP}}(\theta)$
- The critic (value function) loss - $L_{\text{VF}}(\theta)$ like the MSE $(G_{t:N} - V_\phi(s_t))^2$ .
- An entropy bonus $S[\pi_\theta]$

So, the combined objective might look like:

$$
L_{\text{PPO}}(\theta)= -\hat{\mathbb{E}}_t \left[ L_{\text{CLIP}}(\theta) - c_1 L_{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

Where $c_1$ and $c_2$ are coefficients to weight these different terms. We'd then perform gradient descent on this combined loss.

---

**Good thing to ask why then we cannot use Importance Sampling to solve this issue of using replay buffer in on-policy algorithms?**

This question might have already risen in your head. We already seen how IS can be used to reuse data for several learning steps. Then, why we can't extend it to a full replay buffer?

Well, in pure theory IS can be a silver bullet to combine both on-pollicy algorithms and replay buffer, but it fails in practice in a fully off-policy setting for one main reason: exploding variance.

Importance sampling correction involves scaling the update by the ratio of probabilities, $r(\theta)$. In an algorithm like DQN, the replay buffer contains a mix of experiences from many policies, some of which can be very old and different from the current one.

The sampled experience from the buffer might be generated by the policy that was actual thousands of optimization steps before. It is obvious that our current policy might be very, very different from it. The ratio would be huge. A single sample with a massive ratio can completely dominate the gradient for the entire mini-batch, causing a destructive update that collapses the policy. The variance of the gradient estimate becomes unmanageably high, making stable learning impossible.

What if apply PPO clip safe guard? Then, from the same logic, the majority of samples would contribute zero gradient update, because they would probably all be clipped. That would make the learning painfully slow.

While PPO does use IS, it does so in a very controlled way. It only uses data from the most recent policy and, crucially, clips the ratio to prevent it from exploding.

---

## PPO Implementation Details

All right, as we discussed earlier, RL is a very brittle field. Getting PPO to work well often depends on a handful of implementation details that might not be obvious from the main algorithm. These are the "tricks of the trade" that researchers and practitioners have found to stabilize training and lead to better performance. Here are some of the most important ones:

1. **Advantage Normalization**

This is perhaps the most critical detail. Before we use the advantages $\hat{A}_t$ in the actor's loss function, we normalize them. As we'll see in detail #8, this is typically done at the mini-batch level.

**How:** For a batch of advantages, subtract the mean and divide by the standard deviation.

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why:** This stabilizes training significantly. It ensures that the policy updates are not dependent on the arbitrary scale of rewards in an environment. It re-centers the "goodness" of actions around zero for every batch, preventing the agent from being pushed too hard by a single batch that happened to have unusually high or low reward values.

---

2. **Value Function Clipping**

Just like we clip the policy objective, we can also clip the value function objective. This is mentioned in the original PPO paper and helps stabilize the value function, which in turn leads to more stable advantage estimates.

**How:** The critic's loss is usually $(V_\text{predicted} - V_\text{target})^2$. We can clip the predicted value $V_\text{predicted}$ so it doesn't move too far from the value we collected during the rollout, $V_\text{old}$. The new loss becomes $\max\left((V_\text{predicted} - V_\text{target})^2, (V_\text{clipped} - V_\text{target})^2\right)$, where $V_\text{clipped} = V_\text{old} + \text{clip}(V_\text{predicted} - V_\text{old}, -\epsilon, +\epsilon)$.

**Why:** This prevents the value function from changing too drastically in one update phase, which is important because the actor's learning depends heavily on the stability of these value estimates.

---

3. **Global Gradient Clipping**

This is a standard safety rail in deep learning, but it's especially important in RL where strange batches of data can sometimes lead to massive gradients.

**How:** After calculating the gradients during the backward pass (`loss.backward()`) but before the optimizer takes a step (`optimizer.step()`), we clip the overall norm of the gradients for the entire network.

```python
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
```

**Why:** This prevents a single unusual batch from creating an "exploding gradient" that would take a destructively large step and completely wreck the network's weights.

---

4. **Learning Rate Annealing**

Instead of using a fixed learning rate throughout training, it's very common to linearly decay it.

**How:** Start with an initial learning rate (e.g., 3e-4) and decrease it linearly at every update step, such that it reaches zero by the end of training.

**Why:** Intuitively, we want the agent to take large learning steps and explore broadly at the beginning. As it gets closer to a good policy, we want it to take smaller, more precise steps to fine-tune its behavior without overshooting the optimal solution.

---

5. **Network Architecture and Initialization**

The small details of the network itself matter.

- **Activation Functions:** Many successful PPO implementations use `tanh` as the activation function in the hidden layers, as it can help keep network activations within a bounded range.
- **Weight Initialization:** Instead of the default initialization, using Orthogonal Initialization for the network weights is a very popular and effective technique that has been shown to work well for policy and value networks in RL.

---

6. **Avoid Dropout and Batch Normalization**

**How:** Avoid using `nn.Dropout` and `nn.BatchNorm` layers in your actor and critic networks.

**Why:** PPO relies on re-calculating the log probabilities of actions from the collected data. Stochastic layers like Dropout or stateful layers like BatchNorm behave differently between when you collect the data and when you train. This creates a mismatch. The `log_prob` you re-calculate during training might not correspond to the one from the policy that actually took the action, which can break the importance sampling assumption and destabilize training. It's generally safer to rely on other regularization methods like weight decay if needed.

---

7. **Mini-batch Updates**

Instead of performing one large gradient update on the entire batch of rollout data ($N$ steps $\times$ $M$ environments), it is standard practice to loop over the data for several "PPO epochs" and perform updates on smaller mini-batches.

**How:** After collecting the full batch of data, create an array of indices from $0$ to $N \times M - 1$. For each PPO epoch, shuffle these indices. Then, iterate through the shuffled indices, taking out mini-batches of data (e.g., of size 64 or 256) and performing a gradient update on each one.

**Why:** This is a standard deep learning technique. It leads to more stable and efficient learning. The gradient estimates from mini-batches are noisier but provide much more frequent updates, helping the optimizer converge more smoothly than it would with a single, large, and computationally expensive update.

---

8. **Per-Mini-batch Advantage Normalization**

This is a subtle but powerful refinement of detail #1. Instead of normalizing the advantages once over the entire batch of data, you should re-normalize them within each mini-batch just before calculating the loss for that mini-batch.

**How:** Inside your mini-batch loop, take the slice of advantages corresponding to that mini-batch and normalize just that slice.

**Why:** This ensures that the scale of the policy gradient is consistent across all mini-batch updates within an epoch. It prevents a mini-batch that happens to contain outlier advantage values from creating an unusually large or small gradient, further stabilizing the learning process.

---

9. **Separate vs. Shared Networks**

You can structure your Actor and Critic networks in two ways:

- **Shared Network:** A common "base" network (e.g., CNN layers for image input) processes the observation, and then the output splits into two separate "heads" - one that outputs the action probabilities (the actor) and one that outputs the state value (the critic).
- **Separate Networks:** The Actor and Critic are two completely independent neural networks, each with its own set of weights and its own optimizer.

**Which is better?** While a shared network is more parameter-efficient, many high-performing implementations use separate networks. The reason is that the policy and value functions have very different objectives. The policy loss and value loss can sometimes create conflicting gradients that interfere with each other when they have to update a shared set of weights. Using separate networks completely isolates them, which can lead to more stable training, especially in simpler environments where parameter efficiency is less of a concern.

---

With these details, we now have a complete picture of Proximal Policy Optimization, a robust and effective on-policy algorithm.

PPO is considered a state-of-the-art algorithm, even at the time of writing (2025). There are many reasons why PPO became so popular and why it achieves such good results on different problems:

- **Stability**: It is remarkably stable. The clipped surrogate objective prevents the policy from diverging, ensuring that as more data is collected, the policy reliably improves rather than deteriorating.

- **Sample Efficiency**: It's highly sample-efficient for an on-policy algorithm. By using an importance sampling ratio, PPO can train for several epochs on the same batch of data, allowing it to learn more from each experience.

- **Simplicity**: It is relatively simple to implement. Its predecessor, TRPO, was effective but complex. PPO became popular because it offers similar stability benefits with an algorithm that is more straightforward to understand and code from scratch.

- **Scalability**: It scales well with parallel environments. As a successor to A2C, PPO is naturally designed for parallelization, which makes training both faster and more stable by decorrelating the collected data.

So, if you have a problem and you want to apply some RL to see how it goes, then start with PPO, it won't be a mistake.

# Conclusion

Policy gradient algorithms represent a powerful tool in our arsenal. In contrast to value-based methods (RAINBOW), policy gradients learn the optimal policy directly.
