import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates

@jax.jit
def augment_image(image: jnp.ndarray, shift_y: float, shift_x: float) -> jnp.ndarray:
    """
    Shifts a single 28x28 image by (shift_y, shift_x).
    Uses bilinear interpolation (order=1).
    """
    image = image.reshape((28, 28))
    y, x = jnp.meshgrid(jnp.arange(28), jnp.arange(28), indexing="ij")
    y_shifted = jnp.clip(y - shift_y, 0, 27)
    x_shifted = jnp.clip(x - shift_x, 0, 27)
    augmented = map_coordinates(image, [y_shifted, x_shifted], order=1, mode="constant", cval=0.0)
    return augmented.reshape(-1)

@jax.jit
def batch_augment_images(images: jnp.ndarray, shift_y: jnp.ndarray, shift_x: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized augmentation across a batch of images.
    images: (B, 784)
    shift_y, shift_x: (B,)
    Returns: (B, 784)
    """
    return jax.vmap(augment_image)(images, shift_y, shift_x)

def generate_shifts(key, batch_size, max_shift):
    """Generates random y/x shifts for a batch."""
    key_y, key_x = random.split(key)
    shift_y = random.uniform(key_y, (batch_size,), minval=-max_shift, maxval=max_shift)
    shift_x = random.uniform(key_x, (batch_size,), minval=-max_shift, maxval=max_shift)
    return shift_y, shift_x
