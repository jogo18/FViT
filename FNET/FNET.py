import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable


class FourierTransformLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.vmap(jnp.fft.fftn)(x).real


class FeedForwardLayer(nn.Module):
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.d_ff,
                     kernel_init=nn.initializers.normal(2e-2),
                     bias_init=nn.initializers.normal(2e-2),
                     name="intermediate")(x)
        x = nn.gelu(x)
        x = nn.Dense(x.shape[-1],
                     kernel_init=nn.initializers.normal(2e-2),
                     name="output")(x)
        return nn.Dropout(self.dropout_rate)(x, deterministic)


class FNetEncoderBlock(nn.Module):
    fourier_layer: FourierTransformLayer
    ff_layer: FeedForwardLayer

    @nn.compact
    def __call__(self, x, deterministic):
        mixing_output = self.fourier_layer(x)
        x = nn.LayerNorm(1e-12, name="mixing_layer_norm")(x + mixing_output)
        feed_forward_output = self.ff_layer(x, deterministic)
        return nn.LayerNorm(
            1e-12, name="output_layer_norm")(x + feed_forward_output)


class FNetEncoder(nn.Module):
    num_layers: int
    d_model: int
    d_ff: int
    dropout_rate: float

    def setup(self):
        encoder_blocks = []
        for layer in range(self.num_layers):
            encoder_blocks.append(FNetEncoderBlock(
                FourierTransformLayer(),
                FeedForwardLayer(self.d_ff, self.dropout_rate),
                name=f"encoder_{layer}"))
        self.encoder_blocks = encoder_blocks
        self.pooler = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(2e-2),
            name="pooler")

    def __call__(self, x, deterministic):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, deterministic)
        pooled_output = self.pooler(x[:, 0])
        pooled_output = jnp.tanh(pooled_output)
        return x, pooled_output
