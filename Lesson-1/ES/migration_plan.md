# Migration Plan: From VBN to Observation Normalization

## Overview
OpenAI's ES implementation uses observation normalization at the input level, not Virtual Batch Normalization (VBN) throughout the network. This document outlines the steps to migrate from the current VBN implementation to OpenAI's approach.

## Key Differences

### Current Implementation (VBN)
- Virtual Batch Normalization layers after each linear/conv layer
- Fixed reference batch statistics computed once during initialization
- Learnable gamma and beta parameters in each VBN layer
- Complex hooks for capturing activations

### OpenAI Implementation (Observation Normalization)
- Simple input normalization: `(observation - mean) / std`
- Running statistics that update during training
- Clipping normalized values to [-5.0, 5.0]
- No learnable parameters for normalization
- Statistics collected by workers and aggregated by master

## Action Items

### 1. Implement RunningStat Class
Create a class similar to OpenAI's for tracking observation statistics:
```python
class RunningStat:
    def __init__(self, shape, eps=1e-2):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))
```

### 2. Modify Policy Networks

#### Remove VBN Layers
- Remove all VirtualBatchNorm1d and VirtualBatchNorm2d instances
- Remove VBN-related methods: `set_reference_batch`, `get_vbn_stats`, `set_vbn_stats`
- Simplify network architecture to basic layers without normalization

#### Add Input Normalization
- Add observation mean/std parameters to the network
- Normalize inputs before feeding to the network
- Implement clipping to [-5.0, 5.0] range

#### FCNetwork Changes
```python
class FCNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space=64):
        super(FCNetwork, self).__init__()
        self.register_buffer("ob_mean", torch.zeros(obs_space))
        self.register_buffer("ob_std", torch.ones(obs_space))

        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
        )

    def forward(self, x):
        # Normalize input
        x_normalized = (x - self.ob_mean) / self.ob_std
        x_normalized = torch.clamp(x_normalized, -5.0, 5.0)
        return self.network(x_normalized)

    def set_ob_stat(self, mean, std):
        self.ob_mean.copy_(torch.tensor(mean))
        self.ob_std.copy_(torch.tensor(std))
```

### 3. Modify ESAgent Class

#### Remove VBN Initialization
- Remove `_initialize_vbn` method
- Remove `vbn_stats` attribute
- Remove VBN-related parameters from constructor

#### Add Observation Statistics
- Add `ob_stat` attribute using RunningStat
- Implement observation collection during rollouts
- Update statistics after each batch

#### Key Changes:
```python
def __init__(self, ...):
    # Remove VBN parameters
    # Add observation statistics
    self.ob_stat = RunningStat(env.observation_space.shape)
    self.collect_ob_stat = True  # or from config

def play_one_episode(self, seed=None, collect_obs=False):
    # Modify to optionally collect observations
    if collect_obs:
        observations = []
    # ... episode logic ...
    if collect_obs:
        return total_reward, np.array(observations)
    return total_reward
```

### 4. Update Training Loop

#### Statistics Collection
- Workers collect observations during rollouts
- Send observation sum, sum of squares, and count to master
- Master aggregates statistics from all workers

#### Multiprocessing Changes
- Modify `MutantEvaluationParams` to include ob_mean and ob_std
- Update `evaluate_mutant` to use observation normalization
- Remove VBN statistics passing

### 5. Configuration Updates

#### ES_CONFIG Changes
```python
ES_CONFIG = {
    # Remove VBN parameters
    # "use_vbn": True,
    # "vbn_batch_size": 128,

    # Add observation normalization parameters
    "use_ob_norm": True,
    "calc_obstat_prob": 0.01,  # Probability of collecting obs stats
    # ... other parameters ...
}
```

### 6. Backward Compatibility

To maintain compatibility:
1. Add a `normalization_type` parameter: "vbn", "ob_norm", or "none"
2. Keep VBN code but deprecate it
3. Default to "ob_norm" for new experiments

## Implementation Order

1. **Phase 1**: Implement RunningStat class
2. **Phase 2**: Create new policy networks without VBN
3. **Phase 3**: Update agent to support observation normalization
4. **Phase 4**: Modify training loop and multiprocessing
5. **Phase 5**: Update configuration and add compatibility layer
6. **Phase 6**: Test and validate performance

## Testing Strategy

1. Compare normalized observations between implementations
2. Verify statistics convergence
3. Performance comparison on standard environments
4. Ensure multiprocessing works correctly

## Expected Benefits

1. **Simpler Architecture**: Removes complex VBN layers
2. **Better Alignment**: Matches OpenAI's proven approach
3. **Adaptive Normalization**: Statistics update during training
4. **Reduced Memory**: No need to store reference batch
5. **Cleaner Code**: Simpler forward pass without hooks

## Potential Risks

1. Performance may differ from current VBN approach
2. Need to retune hyperparameters
3. Existing saved models won't be compatible
