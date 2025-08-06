import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(FCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
        )
        self._init_weights()
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class CNNNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(CNNNetwork, self).__init__()
        C,H,W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        self._init_weights()

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
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DuelingFC(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(DuelingFC, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    
class DuelingCNN(nn.Module):
    def __init__(self, obs_space, action_space, hidden_space: int = 128):
        super(DuelingCNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )   
        # Pass a zero tensor to get the final flattened shape of the resulting tensor
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1) # A single output V(s)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers and Xavier for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DistributionalFC(nn.Module):
    """Distributional DQN (C51) network for classic control environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51):
        super(DistributionalFC, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space * n_atoms),
        )
        self._init_weights()
        
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        # Reshape to [batch_size, action_space, n_atoms]
        x = x.view(-1, self.action_space, self.n_atoms)
        # Apply log_softmax over the atoms dimension to get log probabilities
        return F.log_softmax(x, dim=-1)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DistributionalCNN(nn.Module):
    """Distributional DQN (C51) network for Atari environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51):
        super(DistributionalCNN, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        C, H, W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Pass a zero tensor to get the final flattened shape
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
            
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space * n_atoms)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        # Extract features from the image
        x = self.backbone(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        # Reshape to [batch_size, action_space, n_atoms]
        x = x.view(-1, self.action_space, self.n_atoms)
        # Apply log_softmax over the atoms dimension
        return F.log_softmax(x, dim=-1)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers and Xavier for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DistributionalDuelingFC(nn.Module):
    """Distributional Dueling DQN network for classic control environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51):
        super(DistributionalDuelingFC, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        self.feature = nn.Sequential(
            nn.Linear(obs_space, hidden_space),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space * n_atoms)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, n_atoms)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x).view(-1, self.action_space, self.n_atoms)
        value = self.value(x).view(-1, 1, self.n_atoms)
        
        # Combine value and advantage streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply log_softmax over the atoms dimension
        return F.log_softmax(q_atoms, dim=-1)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DistributionalDuelingCNN(nn.Module):
    """Distributional Dueling DQN network for Atari environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51):
        super(DistributionalDuelingCNN, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Pass a zero tensor to get the final flattened shape
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space * n_atoms)
        )
        
        self.value = nn.Sequential(
            nn.Linear(feature_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, n_atoms)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        advantage = self.advantage(x).view(-1, self.action_space, self.n_atoms)
        value = self.value(x).view(-1, 1, self.n_atoms)
        
        # Combine value and advantage streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply log_softmax over the atoms dimension
        return F.log_softmax(q_atoms, dim=-1)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers and Xavier for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration via noisy networks"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register buffer for epsilon (noise)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Forward pass with noisy weights and bias"""
        # If in train mode, then use stochastic policy
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else: # If in eval() mode then use deterministic policy
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyFC(nn.Module):
    """Noisy fully-connected network for classic control environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, std_init: float = 0.5):
        super(NoisyFC, self).__init__()
        self.network = nn.Sequential(
            NoisyLinear(obs_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space, std_init),
        )
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyCNN(nn.Module):
    """Noisy CNN network for Atari environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, std_init: float = 0.5):
        super(NoisyCNN, self).__init__()
        C, H, W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate feature size
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.regressor = nn.Sequential(
            NoisyLinear(feature_size, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space, std_init)
        )
        self._init_conv_weights()
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def _init_conv_weights(self):
        """Initialize convolutional layer weights"""
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class NoisyDuelingFC(nn.Module):
    """Noisy Dueling network for classic control environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, std_init: float = 0.5):
        super(NoisyDuelingFC, self).__init__()
        self.feature = nn.Sequential(
            NoisyLinear(obs_space, hidden_space, std_init),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space, std_init)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(hidden_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, 1, std_init)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyDuelingCNN(nn.Module):
    """Noisy Dueling network for Atari environments"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, std_init: float = 0.5):
        super(NoisyDuelingCNN, self).__init__()
        C, H, W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate feature size
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            NoisyLinear(feature_size, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space, std_init)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(feature_size, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, 1, std_init)
        )
        self._init_conv_weights()
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def _init_conv_weights(self):
        """Initialize convolutional layer weights"""
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


# RAINBOW Networks (Noisy + Distributional + Dueling)
class NoisyDistributionalDuelingFC(nn.Module):
    """RAINBOW network for classic control environments combining Noisy, Distributional, and Dueling architectures"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51, std_init: float = 0.5):
        super(NoisyDistributionalDuelingFC, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        self.feature = nn.Sequential(
            NoisyLinear(obs_space, hidden_space, std_init),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space * n_atoms, std_init)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(hidden_space, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, n_atoms, std_init)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        advantage = self.advantage(x).view(-1, self.action_space, self.n_atoms)
        value = self.value(x).view(-1, 1, self.n_atoms)
        
        # Combine value and advantage streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply log_softmax over the atoms dimension
        return F.log_softmax(q_atoms, dim=-1)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyDistributionalDuelingCNN(nn.Module):
    """RAINBOW network for Atari environments combining Noisy, Distributional, and Dueling architectures"""
    def __init__(self, obs_space, action_space, hidden_space: int = 128, n_atoms: int = 51, std_init: float = 0.5):
        super(NoisyDistributionalDuelingCNN, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        
        C, H, W = obs_space
        self.backbone = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate feature size
        with torch.no_grad():
            feature_size = self.backbone(torch.zeros(1, *obs_space)).view(1, -1).size(1)
        
        self.advantage = nn.Sequential(
            NoisyLinear(feature_size, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, action_space * n_atoms, std_init)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(feature_size, hidden_space, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_space, n_atoms, std_init)
        )
        self._init_conv_weights()
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        advantage = self.advantage(x).view(-1, self.action_space, self.n_atoms)
        value = self.value(x).view(-1, 1, self.n_atoms)
        
        # Combine value and advantage streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply log_softmax over the atoms dimension
        return F.log_softmax(q_atoms, dim=-1)
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def _init_conv_weights(self):
        """Initialize convolutional layer weights"""
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)