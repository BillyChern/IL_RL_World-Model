"""Soft Actor-Critic (SAC) for continuous control.

Implements SAC with:
- Twin Q-networks for stability
- Entropy-regularized policy
- Automatic temperature tuning
- Support for hybrid IL+RL loss (HIL-SERL)

Based on: https://github.com/rail-berkeley/serl

Author: Billy Chern (Shichen)
License: MIT
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""

    # Network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # State/action dimensions
    state_dim: int = 14  # Joint positions
    action_dim: int = 14  # Joint commands
    visual_features_dim: int = 512  # From vision encoder

    # SAC hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    alpha_init: float = 0.2  # Initial temperature
    auto_alpha: bool = True  # Automatic temperature tuning

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # HIL-SERL specific
    use_bc_loss: bool = True  # Use behavioral cloning loss
    bc_loss_weight: float = 1.0  # Weight for BC loss
    bc_loss_decay: float = 0.999  # Decay rate for BC loss weight

    # Training
    batch_size: int = 256
    grad_clip: float = 10.0


class MLP(nn.Module):
    """Multi-layer perceptron."""

    hidden_dim: int
    num_layers: int
    output_dim: int
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input [B, input_dim]

        Returns:
            Output [B, output_dim]
        """
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.activation == 'relu':
                x = nn.relu(x)
            elif self.activation == 'tanh':
                x = nn.tanh(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class Actor(nn.Module):
    """Stochastic policy network (actor)."""

    config: SACConfig

    @nn.compact
    def __call__(
        self, state: jnp.ndarray, visual_features: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute action distribution.

        Args:
            state: Proprioceptive state [B, state_dim]
            visual_features: Visual features [B, visual_features_dim]

        Returns:
            mean: Action mean [B, action_dim]
            log_std: Action log std [B, action_dim]
        """
        # Concatenate state and visual features
        x = jnp.concatenate([state, visual_features], axis=-1)

        # MLP backbone
        x = MLP(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_hidden_layers,
            output_dim=self.config.hidden_dim,
            activation='relu',
        )(x)

        # Mean and log_std heads
        mean = nn.Dense(self.config.action_dim)(x)
        log_std = nn.Dense(self.config.action_dim)(x)

        # Clip log_std for numerical stability
        log_std = jnp.clip(log_std, -20, 2)

        return mean, log_std

    def sample_action(
        self,
        state: jnp.ndarray,
        visual_features: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action using reparameterization trick.

        Args:
            state: State [B, state_dim]
            visual_features: Visual features [B, visual_features_dim]
            rng: Random key

        Returns:
            action: Sampled action [B, action_dim]
            log_prob: Log probability [B]
        """
        mean, log_std = self(state, visual_features)
        std = jnp.exp(log_std)

        # Sample using reparameterization
        noise = jax.random.normal(rng, mean.shape)
        action = mean + std * noise

        # Compute log probability
        log_prob = -0.5 * (
            ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)

        # Apply tanh squashing
        action = jnp.tanh(action)

        # Correct log_prob for tanh
        log_prob = log_prob - jnp.sum(
            jnp.log(1 - action ** 2 + 1e-6), axis=-1
        )

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-function networks."""

    config: SACConfig

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        visual_features: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Q-values from both critics.

        Args:
            state: State [B, state_dim]
            visual_features: Visual features [B, visual_features_dim]
            action: Action [B, action_dim]

        Returns:
            q1: Q-value from critic 1 [B, 1]
            q2: Q-value from critic 2 [B, 1]
        """
        # Concatenate inputs
        x = jnp.concatenate([state, visual_features, action], axis=-1)

        # Twin Q-networks
        q1 = MLP(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_hidden_layers,
            output_dim=1,
            activation='relu',
        )(x)

        q2 = MLP(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_hidden_layers,
            output_dim=1,
            activation='relu',
        )(x)

        return q1, q2


class SACAgent:
    """SAC agent with HIL-SERL extensions."""

    def __init__(self, config: SACConfig, rng: jax.random.PRNGKey):
        """Initialize SAC agent.

        Args:
            config: SAC configuration
            rng: Random key
        """
        self.config = config

        # Split random key
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # Initialize actor
        self.actor = Actor(config=config)
        dummy_state = jnp.zeros((1, config.state_dim))
        dummy_visual = jnp.zeros((1, config.visual_features_dim))

        actor_params = self.actor.init(actor_key, dummy_state, dummy_visual)

        # Initialize critics (main and target)
        self.critic = Critic(config=config)
        dummy_action = jnp.zeros((1, config.action_dim))

        critic_params = self.critic.init(
            critic_key, dummy_state, dummy_visual, dummy_action
        )

        # Create training states
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params['params'],
            tx=optax.adam(config.actor_lr),
        )

        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params['params'],
            tx=optax.adam(config.critic_lr),
        )

        # Target critic (for stability)
        self.target_critic_params = critic_params['params']

        # Temperature parameter (log_alpha for numerical stability)
        self.log_alpha = jnp.log(config.alpha_init)

        if config.auto_alpha:
            # Target entropy = -dim(A)
            self.target_entropy = -config.action_dim

            # Alpha optimizer
            self.alpha_optimizer = optax.adam(config.alpha_lr)
            self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)

        # BC loss weight (decays over time)
        self.bc_loss_weight = config.bc_loss_weight

    @jax.jit
    def select_action(
        self,
        state: jnp.ndarray,
        visual_features: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Select action for execution.

        Args:
            state: State [state_dim]
            visual_features: Visual features [visual_features_dim]
            rng: Random key
            deterministic: If True, use mean action

        Returns:
            action: Selected action [action_dim]
        """
        # Add batch dimension
        state = state[None, :]
        visual_features = visual_features[None, :]

        if deterministic:
            mean, _ = self.actor.apply(
                {'params': self.actor_state.params},
                state,
                visual_features,
            )
            action = jnp.tanh(mean)
        else:
            action, _ = self.actor.apply(
                {'params': self.actor_state.params},
                state,
                visual_features,
                rng,
                method=self.actor.sample_action,
            )

        # Remove batch dimension
        return action[0]

    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey,
    ) -> Dict[str, float]:
        """Update agent (actor and critics).

        Args:
            batch: Batch of transitions with keys:
                - states: [B, state_dim]
                - visual_features: [B, visual_features_dim]
                - actions: [B, action_dim]
                - rewards: [B]
                - next_states: [B, state_dim]
                - next_visual_features: [B, visual_features_dim]
                - dones: [B]
                - is_demo: [B] (1 if from demo, 0 if online)
            rng: Random key

        Returns:
            metrics: Training metrics
        """
        # Update critic
        critic_metrics, new_critic_state, new_target_params = self._update_critic(
            self.critic_state,
            self.target_critic_params,
            self.actor_state.params,
            batch,
            rng,
        )

        self.critic_state = new_critic_state
        self.target_critic_params = new_target_params

        # Update actor
        actor_metrics, new_actor_state = self._update_actor(
            self.actor_state,
            self.critic_state.params,
            batch,
            rng,
        )

        self.actor_state = new_actor_state

        # Update temperature (if auto)
        if self.config.auto_alpha:
            alpha_metrics, new_log_alpha, new_alpha_opt_state = self._update_alpha(
                actor_metrics['entropy']
            )

            self.log_alpha = new_log_alpha
            self.alpha_opt_state = new_alpha_opt_state
        else:
            alpha_metrics = {}

        # Decay BC loss weight
        self.bc_loss_weight *= self.config.bc_loss_decay

        # Combine metrics
        metrics = {**critic_metrics, **actor_metrics, **alpha_metrics}
        metrics['bc_loss_weight'] = self.bc_loss_weight

        return metrics

    @staticmethod
    @jax.jit
    def _update_critic(
        critic_state: train_state.TrainState,
        target_params: Dict,
        actor_params: Dict,
        batch: Dict,
        rng: jax.random.PRNGKey,
    ) -> Tuple[Dict, train_state.TrainState, Dict]:
        """Update critic networks."""
        # Implementation details omitted for brevity
        # See full SAC implementation
        raise NotImplementedError("Full implementation in complete codebase")

    @staticmethod
    @jax.jit
    def _update_actor(
        actor_state: train_state.TrainState,
        critic_params: Dict,
        batch: Dict,
        rng: jax.random.PRNGKey,
    ) -> Tuple[Dict, train_state.TrainState]:
        """Update actor network."""
        # Implementation details omitted for brevity
        raise NotImplementedError("Full implementation in complete codebase")

    def _update_alpha(self, entropy: float) -> Tuple[Dict, jnp.ndarray, any]:
        """Update temperature parameter."""
        # Implementation details omitted for brevity
        raise NotImplementedError("Full implementation in complete codebase")
