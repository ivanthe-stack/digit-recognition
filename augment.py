import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates

IMAGE_SHAPE = (28, 28)
CENTER = 13.5  # center of MNIST image (zero-indexed grid)

@jax.jit
def augment_image(image: jnp.ndarray, shift_y: float, shift_x: float,
                  angle: float, zoom: float) -> jnp.ndarray:
    """
    Applies shift, rotation, and zoom to a single 28x28 image.
    Uses bilinear interpolation (order=1).
    """
    image = image.reshape(IMAGE_SHAPE)
    y, x = jnp.meshgrid(jnp.arange(28), jnp.arange(28), indexing="ij")

    # --- translate to center ---
    y_c = y - CENTER
    x_c = x - CENTER

    # --- apply rotation + zoom ---
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    y_t = (y_c * cos_a - x_c * sin_a) / zoom + CENTER - shift_y
    x_t = (y_c * sin_a + x_c * cos_a) / zoom + CENTER - shift_x

    # --- sample with bilinear interpolation ---
    augmented = map_coordinates(
        image, [y_t, x_t],
        order=1, mode="constant", cval=0.0
    )

    return augmented.reshape(-1)

@jax.jit
def batch_augment_images(images: jnp.ndarray,
                         shift_y: jnp.ndarray, shift_x: jnp.ndarray,
                         angles: jnp.ndarray, zooms: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized augmentation across a batch.
    images: (B, 784)
    shift_y/x, angles, zooms: (B,)
    Returns: (B, 784)
    """
    return jax.vmap(augment_image)(images, shift_y, shift_x, angles, zooms)

def generate_augmentation_params(key, batch_size, max_shift=5, max_angle=jnp.pi/12, zoom_range=(0.9, 1.1)):
    """
    Generates random shift, rotation, and zoom parameters.
    - shift: uniform in [-max_shift, max_shift]
    - rotation: uniform in [-max_angle, +max_angle] radians
    - zoom: uniform in [zoom_range[0], zoom_range[1]]
    """
    key_shift, key_angle, key_zoom = random.split(key, 3)

    shift_y = random.uniform(key_shift, (batch_size,), minval=-max_shift, maxval=max_shift)
    shift_x = random.uniform(key_shift, (batch_size,), minval=-max_shift, maxval=max_shift)
    angles = random.uniform(key_angle, (batch_size,), minval=-max_angle, maxval=max_angle)
    zooms = random.uniform(key_zoom, (batch_size,), minval=zoom_range[0], maxval=zoom_range[1])

    return shift_y, shift_x, angles, zooms
