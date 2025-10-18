"""DreamerV3 World Model with RSSM.

Implements the Recurrent State Space Model (RSSM) from DreamerV3 for learning
world dynamics and enabling imagination rollouts.

Based on: https://github.com/danijar/dreamerv3

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
class DreamerConfig:
    """Configuration for DreamerV3 world model."""

    # RSSM dimensions
    deter_dim: int = 4096  # Deterministic state (DreamerV3 default: 4096)
    stoch_dim: int = 32  # Stochastic state classes
    stoch_classes: int = 32  # Number of classes per categorical
    hidden_dim: int = 1024  # MLP hidden dimension

    # Image encoding
    image_encoder_depth: int = 48  # CNN depth multiplier
    image_channels: int = 3
    image_size: Tuple[int, int] = (224, 224)

    # Action/state dimensions
    action_dim: int = 14
    state_dim: int = 14  # Proprioception

    # Training
    learning_rate: float = 1e-4
    grad_clip: float = 1000.0

    # Prediction/dynamics
    imagination_horizon: int = 15  # Steps to imagine ahead
    kl_scale: float = 1.0  # KL loss weight

    # Model ensemble (for uncertainty)
    num_ensemble: int = 1  # Single model for now


class ImageEncoder(nn.Module):
    """CNN image encoder for visual observations."""

    depth: int = 48

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode image to embedding.

        Args:
            x: Image [B, H, W, C]

        Returns:
            Embedding [B, embed_dim]
        """
        # DreamerV3-style CNN
        depth = self.depth

        x = nn.Conv(depth, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv(depth * 2, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv(depth * 4, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv(depth * 8, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        return x


class ImageDecoder(nn.Module):
    """CNN image decoder for reconstruction."""

    depth: int = 48
    output_shape: Tuple[int, int, int] = (224, 224, 3)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Decode embedding to image.

        Args:
            x: Embedding [B, embed_dim]

        Returns:
            Image [B, H, W, C]
        """
        depth = self.depth

        # Project to spatial
        x = nn.Dense(32 * depth * 14 * 14)(x)
        x = x.reshape((x.shape[0], 14, 14, 32 * depth))

        # Transposed convolutions
        x = nn.ConvTranspose(depth * 8, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(depth * 4, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(depth * 2, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(depth, (4, 4), strides=2, padding='SAME')(x)
        x = nn.relu(x)

        # Output layer
        x = nn.Conv(self.output_shape[2], (1, 1))(x)

        return x


class RSSM(nn.Module):
    """Recurrent State Space Model.

    Maintains belief state with:
    - Deterministic state (h): GRU hidden state
    - Stochastic state (z): Categorical distribution
    """

    deter_dim: int = 4096
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 1024
    action_dim: int = 14

    def setup(self):
        """Initialize RSSM components."""
        self.gru_cell = nn.GRUCell(features=self.deter_dim)

        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.stoch_dim * self.stoch_classes),
        ])

        # Posterior: q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.stoch_dim * self.stoch_classes),
        ])

    def initial_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get initial RSSM state.

        Returns:
            (deter_state, stoch_state)
        """
        deter = jnp.zeros((batch_size, self.deter_dim))
        stoch = jnp.zeros((batch_size, self.stoch_dim, self.stoch_classes))
        return deter, stoch

    def observe(
        self,
        embed: jnp.ndarray,
        action: jnp.ndarray,
        deter_state: jnp.ndarray,
        stoch_state: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Observation step (posterior update).

        Args:
            embed: Image embedding [B, embed_dim]
            action: Action [B, action_dim]
            deter_state: Previous deterministic state [B, deter_dim]
            stoch_state: Previous stochastic state [B, stoch_dim, stoch_classes]

        Returns:
            new_deter: New deterministic state
            new_stoch: New stochastic state
            stats: Dict with KL divergence and other stats
        """
        # Flatten stochastic state
        stoch_flat = stoch_state.reshape((stoch_state.shape[0], -1))

        # GRU update: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        gru_input = jnp.concatenate([stoch_flat, action], axis=-1)
        new_deter, _ = self.gru_cell(deter_state, gru_input)

        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(new_deter)
        prior_logits = prior_logits.reshape(
            (prior_logits.shape[0], self.stoch_dim, self.stoch_classes)
        )

        # Posterior: q(z_t | h_t, x_t)
        post_input = jnp.concatenate([new_deter, embed], axis=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.reshape(
            (post_logits.shape[0], self.stoch_dim, self.stoch_classes)
        )

        # Sample from posterior
        new_stoch = jax.nn.softmax(post_logits, axis=-1)

        # Compute KL divergence
        kl = self._kl_divergence(post_logits, prior_logits)

        stats = {
            'kl': kl.mean(),
            'prior_entropy': self._entropy(prior_logits).mean(),
            'post_entropy': self._entropy(post_logits).mean(),
        }

        return new_deter, new_stoch, stats

    def imagine(
        self,
        action: jnp.ndarray,
        deter_state: jnp.ndarray,
        stoch_state: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Imagination step (prior prediction).

        Args:
            action: Action [B, action_dim]
            deter_state: Current deterministic state [B, deter_dim]
            stoch_state: Current stochastic state [B, stoch_dim, stoch_classes]

        Returns:
            new_deter: Predicted deterministic state
            new_stoch: Predicted stochastic state
        """
        # Flatten stochastic state
        stoch_flat = stoch_state.reshape((stoch_state.shape[0], -1))

        # GRU update
        gru_input = jnp.concatenate([stoch_flat, action], axis=-1)
        new_deter, _ = self.gru_cell(deter_state, gru_input)

        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(new_deter)
        prior_logits = prior_logits.reshape(
            (prior_logits.shape[0], self.stoch_dim, self.stoch_classes)
        )

        # Sample from prior
        new_stoch = jax.nn.softmax(prior_logits, axis=-1)

        return new_deter, new_stoch

    def _kl_divergence(
        self, post_logits: jnp.ndarray, prior_logits: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute KL divergence between posterior and prior."""
        post = jax.nn.softmax(post_logits, axis=-1)
        prior = jax.nn.softmax(prior_logits, axis=-1)

        kl = post * (jax.nn.log_softmax(post_logits, axis=-1) -
                     jax.nn.log_softmax(prior_logits, axis=-1))

        return kl.sum(axis=-1)

    def _entropy(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Compute entropy of categorical distribution."""
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -(probs * log_probs).sum(axis=-1)


class RewardPredictor(nn.Module):
    """Predict rewards from RSSM state."""

    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray) -> jnp.ndarray:
        """Predict reward.

        Args:
            deter: Deterministic state [B, deter_dim]
            stoch: Stochastic state [B, stoch_dim, stoch_classes]

        Returns:
            Reward prediction [B, 1]
        """
        stoch_flat = stoch.reshape((stoch.shape[0], -1))
        x = jnp.concatenate([deter, stoch_flat], axis=-1)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x


class DreamerV3(nn.Module):
    """Complete DreamerV3 world model."""

    config: DreamerConfig

    def setup(self):
        """Initialize all components."""
        self.image_encoder = ImageEncoder(depth=self.config.image_encoder_depth)
        self.image_decoder = ImageDecoder(
            depth=self.config.image_encoder_depth,
            output_shape=(self.config.image_size[0],
                         self.config.image_size[1],
                         self.config.image_channels)
        )
        self.rssm = RSSM(
            deter_dim=self.config.deter_dim,
            stoch_dim=self.config.stoch_dim,
            stoch_classes=self.config.stoch_classes,
            hidden_dim=self.config.hidden_dim,
            action_dim=self.config.action_dim,
        )
        self.reward_predictor = RewardPredictor(hidden_dim=self.config.hidden_dim)

    def __call__(
        self,
        images: jnp.ndarray,
        actions: jnp.ndarray,
        deter_state: Optional[jnp.ndarray] = None,
        stoch_state: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """Forward pass for training.

        Args:
            images: Image sequence [B, T, H, W, C]
            actions: Action sequence [B, T, action_dim]
            deter_state: Initial deterministic state (optional)
            stoch_state: Initial stochastic state (optional)

        Returns:
            Dictionary with predictions and losses
        """
        batch_size, seq_len = images.shape[:2]

        # Initialize state if not provided
        if deter_state is None or stoch_state is None:
            deter_state, stoch_state = self.rssm.initial_state(batch_size)

        # Lists to accumulate results
        deter_states = []
        stoch_states = []
        image_preds = []
        reward_preds = []
        kl_losses = []

        # Process sequence
        for t in range(seq_len):
            # Encode current image
            embed = self.image_encoder(images[:, t])

            # RSSM observe step
            deter_state, stoch_state, stats = self.rssm.observe(
                embed, actions[:, t], deter_state, stoch_state
            )

            # Store states
            deter_states.append(deter_state)
            stoch_states.append(stoch_state)
            kl_losses.append(stats['kl'])

            # Predict image
            stoch_flat = stoch_state.reshape((batch_size, -1))
            state_concat = jnp.concatenate([deter_state, stoch_flat], axis=-1)
            image_pred = self.image_decoder(state_concat)
            image_preds.append(image_pred)

            # Predict reward
            reward_pred = self.reward_predictor(deter_state, stoch_state)
            reward_preds.append(reward_pred)

        # Stack results
        image_preds = jnp.stack(image_preds, axis=1)  # [B, T, H, W, C]
        reward_preds = jnp.stack(reward_preds, axis=1)  # [B, T, 1]
        kl_loss = jnp.stack(kl_losses, axis=0).mean()  # Scalar

        return {
            'image_preds': image_preds,
            'reward_preds': reward_preds,
            'kl_loss': kl_loss,
            'deter_state': deter_states[-1],
            'stoch_state': stoch_states[-1],
        }

    def imagine_trajectory(
        self,
        initial_deter: jnp.ndarray,
        initial_stoch: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> Dict:
        """Imagine future trajectory using the world model.

        Args:
            initial_deter: Initial deterministic state [B, deter_dim]
            initial_stoch: Initial stochastic state [B, stoch_dim, stoch_classes]
            actions: Imagined actions [B, H, action_dim]

        Returns:
            Dictionary with imagined states and predictions
        """
        batch_size, horizon = actions.shape[:2]

        deter_state = initial_deter
        stoch_state = initial_stoch

        deter_states = []
        stoch_states = []
        reward_preds = []

        for t in range(horizon):
            # Imagine next state
            deter_state, stoch_state = self.rssm.imagine(
                actions[:, t], deter_state, stoch_state
            )

            deter_states.append(deter_state)
            stoch_states.append(stoch_state)

            # Predict reward
            reward_pred = self.reward_predictor(deter_state, stoch_state)
            reward_preds.append(reward_pred)

        return {
            'deter_states': jnp.stack(deter_states, axis=1),
            'stoch_states': jnp.stack(stoch_states, axis=1),
            'reward_preds': jnp.stack(reward_preds, axis=1),
        }


def create_train_state(
    rng: jax.random.PRNGKey,
    config: DreamerConfig,
    learning_rate: float,
) -> train_state.TrainState:
    """Create training state for DreamerV3.

    Args:
        rng: Random key
        config: Model configuration
        learning_rate: Learning rate

    Returns:
        Training state
    """
    model = DreamerV3(config=config)

    # Initialize with dummy data
    dummy_images = jnp.zeros((1, 10, config.image_size[0], config.image_size[1], 3))
    dummy_actions = jnp.zeros((1, 10, config.action_dim))

    variables = model.init(rng, dummy_images, dummy_actions)

    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(learning_rate),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )


def compute_world_model_loss(
    state: train_state.TrainState,
    images: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    config: DreamerConfig,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute world model training loss.

    Args:
        state: Training state
        images: Image sequence [B, T, H, W, C]
        actions: Action sequence [B, T, action_dim]
        rewards: Reward sequence [B, T]
        config: Model configuration

    Returns:
        loss: Total loss
        metrics: Dictionary of metrics
    """
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, images, actions)

        # Image reconstruction loss
        image_loss = jnp.square(outputs['image_preds'] - images).mean()

        # Reward prediction loss
        reward_loss = jnp.square(
            outputs['reward_preds'].squeeze(-1) - rewards
        ).mean()

        # KL loss (with free nats)
        kl_loss = jnp.maximum(outputs['kl_loss'], 1.0)  # Free nats = 1.0

        # Total loss
        total_loss = image_loss + reward_loss + config.kl_scale * kl_loss

        metrics = {
            'total_loss': total_loss,
            'image_loss': image_loss,
            'reward_loss': reward_loss,
            'kl_loss': outputs['kl_loss'],
        }

        return total_loss, metrics

    return jax.value_and_grad(loss_fn, has_aux=True)(state.params)


@jax.jit
def train_step(
    state: train_state.TrainState,
    images: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    config: DreamerConfig,
) -> Tuple[train_state.TrainState, Dict]:
    """Single training step.

    Args:
        state: Training state
        images: Batch of image sequences
        actions: Batch of action sequences
        rewards: Batch of reward sequences
        config: Model configuration

    Returns:
        new_state: Updated training state
        metrics: Training metrics
    """
    (loss, metrics), grads = compute_world_model_loss(
        state, images, actions, rewards, config
    )

    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics
