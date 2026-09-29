"""
MA-LSTM Policy for MAPPO.

This module implements an LSTM actor and centralized critic for the MA-LSTM-PPO
algorithm, compatible with the existing MAPPO codebase.

Actor (per-agent, recurrent):
- Input: agent observation vector o_i (obs_dim)
- Dense -> ReLU -> LSTM (hidden H) -> Dense -> action mean mu (action_dim)
- Learnable log_std parameter (diagonal Gaussian)
- Output: continuous v_des = [v_x, v_y, v_z, thrust]
- Optional tanh to bound outputs

Centralized Critic:
- Input: share_obs (global state: concatenated positions, velocities of all agents)
- Dense -> ReLU -> (optional LSTM) -> Dense -> scalar value V(s)

I/O contract (must match MAPPO runner):
- act(obs, share_obs, rnn_states, deterministic) -> (value, action, log_probs, rnn_states_out)
- evaluate_actions(obs, share_obs, actions, rnn_states) -> (values, log_probs, entropy)

"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union

from onpolicy.algorithms.utils.util import init, check
from onpolicy.utils.util import get_shape_from_obs_space


def run_lstm(
    lstm: nn.LSTM,
    features: torch.Tensor,
    rnn_states: torch.Tensor,
    masks: torch.Tensor,
    num_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run an LSTM over either a single step or a batch of sequence chunks.

    Buffer layout for the hidden state is (N, 2 * num_layers, H): h first, then c.

    - Rollout (single step): features (N, F), rnn_states (N, 2L, H), masks (N, 1).
    - Training (recurrent_generator): features (T * N, F) flattened time-major as
      produced by `_flatten(T, N, x)`, rnn_states (N, 2L, H) holding the state at the
      start of each chunk, masks (T * N, 1). The LSTM is unrolled over T and the
      state is reset wherever masks == 0 (start of a new episode inside the chunk).

    Returns:
        outputs: (T * N, H) (or (N, H) for a single step)
        rnn_states_out: (N, 2L, H) state after the last processed step
    """
    n = rnn_states.shape[0]
    t_len = features.shape[0] // n
    h = rnn_states[:, :num_layers, :].transpose(0, 1).contiguous()  # (L, N, H)
    c = rnn_states[:, num_layers:, :].transpose(0, 1).contiguous()
    x = features.reshape(t_len, n, -1)
    m = masks.reshape(t_len, n, 1)

    outputs = []
    for t in range(t_len):
        mask_t = m[t].unsqueeze(0)  # (1, N, 1)
        h = h * mask_t
        c = c * mask_t
        out, (h, c) = lstm(x[t].unsqueeze(1), (h, c))
        outputs.append(out.squeeze(1))
    outputs = torch.stack(outputs, dim=0).reshape(t_len * n, -1)
    rnn_states_out = torch.cat([h, c], dim=0).transpose(0, 1)  # (N, 2L, H)
    return outputs, rnn_states_out


def init_weights(module, gain=np.sqrt(2), bias=0.0):
    """Initialize weights with orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    return module


class LSTMActor(nn.Module):
    """
    LSTM-based Actor network for MA-LSTM-PPO.
    
    - Input: agent observation vector o_i (obs_dim)
    - Dense -> ReLU -> LSTM (hidden H) -> Dense -> action mean mu (action_dim)
    - Learnable log_std parameter (diagonal Gaussian)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        use_orthogonal: bool = True,
        use_tanh_output: bool = True,
        action_scale: float = 1.0,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize LSTM Actor.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (4 for v_des = [vx, vy, vz, throttle])
            hidden_size: Hidden layer size for MLP
            lstm_hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            use_orthogonal: Whether to use orthogonal initialization
            use_tanh_output: Whether to apply tanh to bound actions
            action_scale: Scale factor for actions after tanh
            log_std_init: Initial value for log standard deviation
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            device: Torch device
        """
        super(LSTMActor, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_tanh_output = use_tanh_output
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Observation normalization
        self.obs_norm = nn.LayerNorm(obs_dim)
        
        # Input feature extraction: Dense -> Tanh -> LayerNorm
        # "tanh activation", "layer normalization"
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
        )
        
        # LSTM layer
        # LSTM with hidden_size=256
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        
        # LayerNorm after LSTM 
        self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
        
        # Output layer: action mean
        self.action_mean = nn.Linear(lstm_hidden_size, action_dim)
        
        # Learnable log_std parameter
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Initialize weights
        self._init_weights(use_orthogonal)
        
        self.to(device)
    
    def _init_weights(self, use_orthogonal: bool):
        """Initialize network weights."""
        gain = np.sqrt(2)
        for module in self.fc1:
            if isinstance(module, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(module.weight, gain)
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param) if use_orthogonal else nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Action mean with smaller gain for stability
        if use_orthogonal:
            nn.init.orthogonal_(self.action_mean.weight, 0.01)
        else:
            nn.init.xavier_uniform_(self.action_mean.weight)
        nn.init.constant_(self.action_mean.bias, 0)
    
    def forward(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observations (batch, obs_dim)
            rnn_states: LSTM hidden states (2, num_layers, batch, lstm_hidden_size)
            masks: Mask for resetting RNN states (batch, 1)
            
        Returns:
            action_mean: Mean of action distribution (batch, action_dim)
            action_std: Std of action distribution (batch, action_dim)
            rnn_states_out: Updated LSTM hidden states
        """
        # Feature extraction: obs_norm -> fc1 (Dense->Tanh->LayerNorm)
        obs_normed = self.obs_norm(obs)
        features = self.fc1(obs_normed)  # (batch, hidden_size)

        # Buffer stores (N, recurrent_N=2, hidden): index 0 = h, index 1 = c.
        # During training obs is (T * N, obs_dim) and the LSTM is unrolled over T.
        lstm_out, rnn_states_out = run_lstm(
            self.lstm, features, rnn_states, masks, self.num_lstm_layers
        )

        # Output with LayerNorm after LSTM
        lstm_out = self.lstm_norm(lstm_out)
        action_mean = self.action_mean(lstm_out)  # (batch, action_dim)

        # Apply tanh for bounded actions
        if self.use_tanh_output:
            action_mean = torch.tanh(action_mean) * self.action_scale

        # Get std from learnable log_std
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std).expand_as(action_mean)

        return action_mean, action_std, rnn_states_out
    
    def sample(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            obs: Observations
            rnn_states: LSTM hidden states
            masks: Mask for resetting RNN states
            deterministic: If True, return mean action
            
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            rnn_states_out: Updated hidden states
        """
        action_mean, action_std, rnn_states_out = self.forward(obs, rnn_states, masks)
        
        if deterministic:
            actions = action_mean
            # Compute log prob of mean action
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        return actions, log_probs, rnn_states_out
    
    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            rnn_states: LSTM hidden states
            masks: Mask for resetting RNN states
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        action_mean, action_std, _ = self.forward(obs, rnn_states, masks)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized Critic network for MA-LSTM-PPO.
    
    - Input: share_obs (joint observations from all UAVs)
    - LayerNorm(obs) -> Dense -> Tanh -> LayerNorm -> LSTM -> LayerNorm -> Dense -> V(s)
    """
    
    def __init__(
        self,
        share_obs_dim: int,
        hidden_size: int = 64,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        use_lstm: bool = True,
        use_orthogonal: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize Centralized Critic.
        
        Args:
            share_obs_dim: Shared observation dimension
            hidden_size: Hidden layer size for MLP
            lstm_hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            use_lstm: Whether to use LSTM in critic
            use_orthogonal: Whether to use orthogonal initialization
            device: Torch device
        """
        super(CentralizedCritic, self).__init__()
        
        self.share_obs_dim = share_obs_dim
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_lstm = use_lstm
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Observation normalization
        self.obs_norm = nn.LayerNorm(share_obs_dim)
        
        # Feature extraction: Dense -> Tanh -> LayerNorm
        # one dense layer, one LSTM layer, and one dense layer"
        # "tanh activation", "layer normalization"
        self.fc1 = nn.Sequential(
            nn.Linear(share_obs_dim, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
        )
        
        # Optional LSTM layer
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            )
            # LayerNorm after LSTM
            self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
            value_input_size = lstm_hidden_size
        else:
            self.lstm = None
            self.lstm_norm = None
            value_input_size = hidden_size
        
        # Value output
        self.value_out = nn.Linear(value_input_size, 1)
        
        # Initialize weights
        self._init_weights(use_orthogonal)
        
        self.to(device)
    
    def _init_weights(self, use_orthogonal: bool):
        """Initialize network weights."""
        gain = np.sqrt(2)
        for module in self.fc1:
            if isinstance(module, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(module.weight, gain)
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        if self.lstm is not None:
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param) if use_orthogonal else nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        
        if use_orthogonal:
            nn.init.orthogonal_(self.value_out.weight, 1.0)
        else:
            nn.init.xavier_uniform_(self.value_out.weight)
        nn.init.constant_(self.value_out.bias, 0)
    
    def forward(
        self,
        share_obs: torch.Tensor,
        rnn_states: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through critic network.
        
        Args:
            share_obs: Shared observations (batch, share_obs_dim)
            rnn_states: LSTM hidden states (optional)
            masks: Mask for resetting RNN states (optional)
            
        Returns:
            values: State values (batch, 1)
            rnn_states_out: Updated LSTM hidden states (or None if not using LSTM)
        """
        # Feature extraction: obs_norm -> fc1 (Dense->Tanh->LayerNorm)
        share_obs_normed = self.obs_norm(share_obs)
        features = self.fc1(share_obs_normed)  # (batch, hidden_size)
        
        if self.use_lstm and self.lstm is not None:
            if rnn_states is None or masks is None:
                batch_size = share_obs.shape[0]
                rnn_states = torch.zeros(
                    batch_size, 2 * self.num_lstm_layers, self.lstm_hidden_size
                ).to(**self.tpdv)
                masks = torch.ones(batch_size, 1).to(**self.tpdv)

            # Buffer format: (N, recurrent_N=2, hidden), h first then c.
            lstm_out, rnn_states_out = run_lstm(
                self.lstm, features, rnn_states, masks, self.num_lstm_layers
            )
            # Apply LayerNorm after LSTM
            lstm_out = self.lstm_norm(lstm_out)

            values = self.value_out(lstm_out)
        else:
            values = self.value_out(features)
            rnn_states_out = None
        
        return values, rnn_states_out


class MA_LSTM_Policy(nn.Module):
    """
    MA-LSTM Policy combining LSTM Actor and Centralized Critic.
    
    This class provides the MAPPO-compatible interface for training and inference.
    
    - act(obs, share_obs, rnn_states, deterministic) -> (value, action, log_probs, rnn_states_out)
    - evaluate_actions(obs, share_obs, actions, rnn_states) -> (values, log_probs, entropy)
    """
    
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        act_space,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize MA-LSTM Policy.
        
        Args:
            args: Arguments namespace with configuration
            obs_space: Observation space
            share_obs_space: Shared observation space (for centralized critic)
            act_space: Action space
            device: Torch device
        """
        super(MA_LSTM_Policy, self).__init__()
        
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Get dimensions from spaces
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(share_obs_space)[0]
        
        # Action space handling
        if hasattr(act_space, 'shape'):
            self.action_dim = act_space.shape[0]
        elif hasattr(act_space, 'n'):
            self.action_dim = act_space.n
        else:
            self.action_dim = 4  # Default for v_des
        
        # Get hyperparameters from args
        hidden_size = getattr(args, 'hidden_size', 64)
        use_orthogonal = getattr(args, 'use_orthogonal', True)
        recurrent_N = getattr(args, 'recurrent_N', 1)
        
        # Learning rates
        self.lr = getattr(args, 'lr', 5e-4)
        self.critic_lr = getattr(args, 'critic_lr', 5e-4)
        self.opti_eps = getattr(args, 'opti_eps', 1e-5)
        self.weight_decay = getattr(args, 'weight_decay', 0)
        
        # Create actor network
        # Note: num_lstm_layers=1 always; recurrent_N=2 is only for buffer (stores h and c)
        self.actor = LSTMActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            lstm_hidden_size=hidden_size,
            num_lstm_layers=1,
            use_orthogonal=use_orthogonal,
            use_tanh_output=True,
            device=device,
        )
        
        # Create critic network
        self.critic = CentralizedCritic(
            share_obs_dim=self.share_obs_dim,
            hidden_size=hidden_size,
            lstm_hidden_size=hidden_size,
            num_lstm_layers=1,
            use_lstm=True,
            use_orthogonal=use_orthogonal,
            device=device,
        )
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        
        # Store args for compatibility
        self._hidden_size = hidden_size
        self._recurrent_N = recurrent_N
    
    def get_actions(
        self,
        cent_obs: np.ndarray,
        obs: np.ndarray,
        rnn_states_actor: np.ndarray,
        rnn_states_critic: np.ndarray,
        masks: np.ndarray,
        available_actions: np.ndarray = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute actions and value predictions.
        
        Args:
            cent_obs: Centralized observations for critic (batch, share_obs_dim)
            obs: Local observations for actor (batch, obs_dim)
            rnn_states_actor: Actor RNN states
            rnn_states_critic: Critic RNN states
            masks: RNN masks
            available_actions: Available actions mask (unused for continuous)
            deterministic: Whether to use deterministic actions
            
        Returns:
            values: Value predictions
            actions: Sampled actions
            action_log_probs: Log probabilities of actions
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
        """
        # Convert inputs to tensors
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Get actions from actor
        actions, action_log_probs, rnn_states_actor_new = self.actor.sample(
            obs, rnn_states_actor, masks, deterministic
        )
        
        # Get values from critic
        values, rnn_states_critic_new = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor_new, rnn_states_critic_new
    
    def get_values(
        self,
        cent_obs: np.ndarray,
        rnn_states_critic: np.ndarray,
        masks: np.ndarray,
    ) -> torch.Tensor:
        """
        Get value predictions.
        
        Args:
            cent_obs: Centralized observations
            rnn_states_critic: Critic RNN states
            masks: RNN masks
            
        Returns:
            values: Value predictions
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(
        self,
        cent_obs: np.ndarray,
        obs: np.ndarray,
        rnn_states_actor: np.ndarray,
        rnn_states_critic: np.ndarray,
        action: np.ndarray,
        masks: np.ndarray,
        available_actions: np.ndarray = None,
        active_masks: np.ndarray = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        
        Args:
            cent_obs: Centralized observations
            obs: Local observations
            rnn_states_actor: Actor RNN states
            rnn_states_critic: Critic RNN states
            action: Actions to evaluate
            masks: RNN masks
            available_actions: Available actions mask (unused)
            active_masks: Active agent masks (unused)
            
        Returns:
            values: Value predictions
            action_log_probs: Log probabilities of given actions
            dist_entropy: Distribution entropy
        """
        # Convert inputs to tensors
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Evaluate actions with actor
        action_log_probs, dist_entropy = self.actor.evaluate(
            obs, action, rnn_states_actor, masks
        )
        
        # Get values from critic
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
    
    def act(
        self,
        obs: np.ndarray,
        rnn_states_actor: np.ndarray,
        masks: np.ndarray,
        available_actions: np.ndarray = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actions for execution (no value computation).
        
        Args:
            obs: Local observations
            rnn_states_actor: Actor RNN states
            masks: RNN masks
            available_actions: Available actions mask (unused)
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Sampled actions
            rnn_states_actor: Updated RNN states
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        actions, _, rnn_states_actor_new = self.actor.sample(
            obs, rnn_states_actor, masks, deterministic
        )
        
        return actions, rnn_states_actor_new
    
    def lr_decay(self, episode: int, episodes: int):
        """Decay learning rates linearly."""
        from onpolicy.utils.util import update_linear_schedule
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)


def get_ma_lstm_policy_class():
    """Return the MA_LSTM_Policy class for factory use."""
    return MA_LSTM_Policy
