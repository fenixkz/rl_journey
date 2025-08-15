import numpy as np
import torch
import torch.nn as nn


class VirtualBatchNorm1d(nn.Module):
    """
    Virtual Batch Normalization for 1D inputs (fully connected layers).
    Uses a reference batch to compute normalization statistics, making the network
    more sensitive to parameter perturbations during evolution.
    """

    def __init__(self, num_features, eps=1e-5):
        super(VirtualBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Register buffers for reference batch statistics
        self.register_buffer("ref_mean", torch.zeros(num_features))
        self.register_buffer("ref_var", torch.ones(num_features))
        self.register_buffer("initialized", torch.tensor(False))

    def set_reference_stats(self, mean, var):
        """Set statistics from reference batch"""
        with torch.no_grad():
            self.ref_mean.copy_(mean)
            self.ref_var.copy_(var)
            self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            # If not initialized, use current batch statistics (fallback)
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
        else:
            # Use ONLY reference batch statistics for normalization
            batch_mean = self.ref_mean
            batch_var = self.ref_var

        # Normalize using reference statistics
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        # Scale and shift
        return self.gamma * x_normalized + self.beta


class VirtualBatchNorm2d(nn.Module):
    """
    Virtual Batch Normalization for 2D inputs (convolutional layers).
    Uses a reference batch to compute normalization statistics.
    """

    def __init__(self, num_features, eps=1e-5):
        super(VirtualBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Register buffers for reference batch statistics
        self.register_buffer("ref_mean", torch.zeros(num_features))
        self.register_buffer("ref_var", torch.ones(num_features))
        self.register_buffer("initialized", torch.tensor(False))

    def set_reference_stats(self, mean, var):
        """Set statistics from reference batch"""
        with torch.no_grad():
            self.ref_mean.copy_(mean)
            self.ref_var.copy_(var)
            self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            # If not initialized, use current batch statistics (fallback)
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)
        else:
            # Use ONLY reference batch statistics for normalization
            batch_mean = self.ref_mean
            batch_var = self.ref_var

        # Reshape for broadcasting
        batch_mean = batch_mean.view(1, -1, 1, 1)
        batch_var = batch_var.view(1, -1, 1, 1)

        # Normalize using reference statistics
        x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        # Scale and shift (reshape gamma and beta for broadcasting)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        return gamma * x_normalized + beta


def normc_initializer(std: float = 1.0):
    def _initializer(tensor: torch.Tensor):
        out = np.random.randn(*tensor.shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        tensor.data.copy_(torch.from_numpy(out))

    return _initializer


class FCNetwork(nn.Module):
    """
    A fully-connected neural network. This is our policy, π(θ).
    Given the state, returns action logits for Evolution Strategies.
    Now includes Virtual Batch Normalization for better exploration.
    """

    def __init__(self, obs_space, action_space, hidden_space: int = 64, use_vbn: bool = True):
        super(FCNetwork, self).__init__()
        self.use_vbn = use_vbn

        if use_vbn:
            self.vbn1 = VirtualBatchNorm1d(hidden_space)
            self.vbn2 = VirtualBatchNorm1d(hidden_space)

            self.network = nn.Sequential(
                nn.Linear(obs_space, hidden_space),
                self.vbn1,
                nn.ReLU(),
                nn.Linear(hidden_space, hidden_space),
                self.vbn2,
                nn.ReLU(),
                nn.Linear(hidden_space, action_space),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(obs_space, hidden_space),
                nn.ReLU(),
                nn.Linear(hidden_space, hidden_space),
                nn.ReLU(),
                nn.Linear(hidden_space, action_space),
            )
        self._init_weights()

    def set_reference_batch(self, reference_states):
        """Process reference batch through network to set VBN statistics using hooks"""
        if not self.use_vbn:
            return

        # Dictionary to store intermediate activations
        activations = {}

        # Define hook function to capture activations
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output

            return hook

        # Register hooks on VBN layers
        hooks = []
        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2}
        for name, layer in vbn_layers.items():
            handle = layer.register_forward_hook(get_activation(name))
            hooks.append(handle)

        # Forward pass with reference batch
        with torch.no_grad():
            _ = self.forward(reference_states)

        # Set statistics for each VBN layer
        for name, layer in vbn_layers.items():
            if name in activations:
                activation = activations[name]
                mean = activation.mean(dim=0)
                var = activation.var(dim=0, unbiased=False)
                layer.set_reference_stats(mean, var)

        # Remove hooks
        for handle in hooks:
            handle.remove()

    def get_vbn_stats(self):
        """Extract VBN statistics to pass to workers"""
        if not self.use_vbn:
            return None

        stats = {}
        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2}
        for name, layer in vbn_layers.items():
            stats[name] = {"mean": layer.ref_mean.cpu(), "var": layer.ref_var.cpu()}
        return stats

    def set_vbn_stats(self, stats):
        """Set VBN statistics directly (for workers)"""
        if not self.use_vbn or stats is None:
            return

        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2}
        for name, layer in vbn_layers.items():
            if name in stats:
                layer.set_reference_stats(
                    stats[name]["mean"].to(layer.ref_mean.device), stats[name]["var"].to(layer.ref_var.device)
                )

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                normc_initializer(1.0)(
                    module.weight
                )  # Apply NormC to weights, found to be better for RL (idk why, god's benevolence)
                nn.init.constant_(module.bias, 0.0)  # Keep biases at zero


class CNNNetwork(nn.Module):
    """
    A convolutional neural network for environments with states as images.
    This is our policy, π(θ).
    Given the state, returns action logits for Evolution Strategies.
    Now includes Virtual Batch Normalization for better exploration.
    """

    def __init__(self, obs_space, action_space, hidden_space: int = 64, use_vbn: bool = True):
        super(CNNNetwork, self).__init__()
        self.use_vbn = use_vbn
        C, H, W = obs_space

        if use_vbn:
            self.vbn1 = VirtualBatchNorm2d(32)
            self.vbn2 = VirtualBatchNorm2d(64)
            self.vbn3 = VirtualBatchNorm2d(64)
            self.vbn4 = VirtualBatchNorm1d(hidden_space)

            self.backbone = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4),
                self.vbn1,
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                self.vbn2,
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                self.vbn3,
                nn.ReLU(),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )

        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)

        if use_vbn:
            self.regressor = nn.Sequential(
                nn.Linear(feature_size, hidden_space), self.vbn4, nn.ReLU(), nn.Linear(hidden_space, action_space)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(feature_size, hidden_space), nn.ReLU(), nn.Linear(hidden_space, action_space)
            )
        self._init_weights()

    def set_reference_batch(self, reference_states):
        """Process reference batch through network to set VBN statistics using hooks"""
        if not self.use_vbn:
            return

        # Dictionary to store intermediate activations
        activations = {}

        # Define hook function to capture activations
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output

            return hook

        # Register hooks on VBN layers
        hooks = []
        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2, "vbn3": self.vbn3, "vbn4": self.vbn4}

        for name, layer in vbn_layers.items():
            handle = layer.register_forward_hook(get_activation(name))
            hooks.append(handle)

        # Forward pass with reference batch
        with torch.no_grad():
            _ = self.forward(reference_states)

        # Set statistics for each VBN layer
        for name, layer in vbn_layers.items():
            if name in activations:
                activation = activations[name]
                # For Conv2d layers (vbn1-3)
                if name in ["vbn1", "vbn2", "vbn3"]:
                    mean = activation.mean(dim=[0, 2, 3])
                    var = activation.var(dim=[0, 2, 3], unbiased=False)
                # For Linear layer (vbn4)
                else:
                    mean = activation.mean(dim=0)
                    var = activation.var(dim=0, unbiased=False)
                layer.set_reference_stats(mean, var)

        # Remove hooks
        for handle in hooks:
            handle.remove()

    def get_vbn_stats(self):
        """Extract VBN statistics to pass to workers"""
        if not self.use_vbn:
            return None

        stats = {}
        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2, "vbn3": self.vbn3, "vbn4": self.vbn4}
        for name, layer in vbn_layers.items():
            stats[name] = {"mean": layer.ref_mean.cpu(), "var": layer.ref_var.cpu()}
        return stats

    def set_vbn_stats(self, stats):
        """Set VBN statistics directly (for workers)"""
        if not self.use_vbn or stats is None:
            return

        vbn_layers = {"vbn1": self.vbn1, "vbn2": self.vbn2, "vbn3": self.vbn3, "vbn4": self.vbn4}
        for name, layer in vbn_layers.items():
            if name in stats:
                layer.set_reference_stats(
                    stats[name]["mean"].to(layer.ref_mean.device), stats[name]["var"].to(layer.ref_var.device)
                )

    def forward(self, x: torch.Tensor):
        # Extract features from the image
        x = self.backbone(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return self.regressor(x)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers and Xavier for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                normc_initializer(1.0)(
                    module.weight
                )  # Apply NormC to weights, found to be better for RL (idk why, god's benevolence)
                nn.init.constant_(module.bias, 0.0)  # Keep biases at zero
