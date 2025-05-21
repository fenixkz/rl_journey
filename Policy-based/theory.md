# Policy based learning

We have studied a very big chapter of Reinforcement Learning where we were using so-called Value based learning. This name comes from the fact that our learning is basically comes to the estimation of $Q(s,a)$. Using our estimation we were able to derive a policy (greedy policy), where for any given state $s$ we were choosing action corresponding to the highest $Q(s,a)$ value. So, although we were implictly learning a policy we were not learning it directly. This chapter of Reinforcement learning explores how we can learn a policy directly.


## Theory 

Let's recall our policy $\pi(a | s)$ is a probability distribution over all possible actions we can take from state $s$. How are we able to learn it? 

